"""
miniServe — Inference Engine
Loads the LLM model and runs inference with KV-cache support.

This is the core compute module. It handles:
1. Loading the GPT-2 model and tokenizer from HuggingFace
2. Single-request inference with timing
3. Batched inference with dynamic padding
4. KV-cache integration for multi-turn speedup

KEY CONCEPTS:
- Left-padding: GPT-2 is decoder-only, so we pad from the LEFT
  to keep the generation position aligned across the batch.
- attention_mask: Tells the model which tokens are real (1) vs padding (0).
- past_key_values: The KV-cache from HuggingFace's generate() method.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from server.kv_cache import KVCacheManager
from server.sequence import Sequence


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from a single generation."""
    generated_text: str
    prompt: str
    tokens_generated: int
    inference_time_ms: float
    from_cache: bool = False
    request_id: Optional[str] = None


class InferenceEngine:
    """
    Core inference engine for LLM text generation.
    
    Handles model loading, tokenization, batched generation,
    and KV-cache management.
    
    Usage:
        engine = InferenceEngine()
        result = engine.generate_single("Once upon a time")
        print(result.generated_text)
        
        # Batched:
        results = engine.generate_batch(["Hello", "World"], max_tokens=50)
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        kv_cache_max: int = None,
    ):
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or config.DEVICE
        self.kv_cache = KVCacheManager(max_entries=kv_cache_max or config.KV_CACHE_MAX_ENTRIES)

        logger.info(f"Loading model '{self.model_name}' on device '{self.device}'...")
        load_start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # GPT-2 doesn't have a pad token — set it to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Set padding side to LEFT for decoder-only batched generation
        self.tokenizer.padding_side = "left"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,  # CPU needs float32
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode (disables dropout)

        load_time = time.time() - load_start
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)

        logger.info(
            f"Model loaded in {load_time:.1f}s | "
            f"Parameters: {param_count:,} | "
            f"Size: {model_size_mb:.0f}MB | "
            f"Device: {self.device}"
        )

    def generate_single(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        request_id: str = None,
    ) -> GenerationResult:
        """
        Generate text for a single prompt.
        
        Args:
            prompt: Input text to continue generating from.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            request_id: Optional ID for KV-cache reuse.
            
        Returns:
            GenerationResult with generated text and metrics.
        """
        results = self.generate_batch(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            request_ids=[request_id] if request_id else None,
        )
        return results[0]

    @torch.no_grad()  # Disable gradient computation for inference
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = None,
        temperature: float = None,
        request_ids: list[str] = None,
    ) -> list[GenerationResult]:
        """
        Generate text for a batch of prompts.
        
        This is where dynamic batching pays off:
        - Multiple prompts are tokenized and padded to the same length
        - The model processes them in parallel (even on CPU, there's benefit
          from BLAS-level batching)
        - Each result is returned to its respective client
        
        Args:
            prompts: List of input texts.
            max_tokens: Max new tokens per prompt.
            temperature: Sampling temperature.
            request_ids: Optional list of IDs for KV-cache.
            
        Returns:
            List of GenerationResult, one per prompt.
        """
        max_tokens = max_tokens or config.MAX_NEW_TOKENS
        temperature = temperature or config.TEMPERATURE
        batch_size = len(prompts)

        logger.info(f"Generating batch of {batch_size} prompts (max_tokens={max_tokens})")

        start_time = time.time()

        # ─── Check KV-Cache for existing states ──────────────────────────
        past_key_values = None
        from_cache = False
        if request_ids and len(request_ids) == 1:
            # KV-cache reuse only works for single requests (same sequence)
            cache_entry = self.kv_cache.get(request_ids[0])
            if cache_entry is not None:
                past_key_values = cache_entry.past_key_values
                from_cache = True
                logger.debug(f"Using cached KV-state for {request_ids[0]}")

        # ─── Tokenize with padding ───────────────────────────────────────
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,         # Pad shorter sequences to match longest
            truncation=True,
            max_length=512,       # GPT-2 context window
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_lengths = attention_mask.sum(dim=1).tolist()  # Actual token counts per prompt

        # ─── Generate ────────────────────────────────────────────────────
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_k": config.TOP_K,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,  # Enable KV-cache within generation
        }

        if past_key_values is not None:
            generation_kwargs["past_key_values"] = past_key_values

        outputs = self.model.generate(**generation_kwargs)

        inference_time_ms = (time.time() - start_time) * 1000

        # ─── Decode results ──────────────────────────────────────────────
        results = []
        for i in range(batch_size):
            # Extract only the NEW tokens (skip the input prompt tokens)
            prompt_length = input_ids.shape[1]  # Padded length
            new_token_ids = outputs[i][prompt_length:]
            
            generated_text = self.tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
            )
            tokens_generated = len(new_token_ids)

            result = GenerationResult(
                generated_text=generated_text,
                prompt=prompts[i],
                tokens_generated=tokens_generated,
                inference_time_ms=round(inference_time_ms, 2),
                from_cache=from_cache,
                request_id=request_ids[i] if request_ids else None,
            )
            results.append(result)

        # ─── Store KV-Cache for single requests ──────────────────────────
        if request_ids and len(request_ids) == 1:
            # For single-request batches, we can cache the KV-state
            # For multi-request batches, KV-cache per-sequence is complex
            # and not supported in this implementation
            self.kv_cache.put(
                request_ids[0],
                None,  # We don't extract past_key_values from generate() output
                prompt_tokens=input_lengths[0],
            )

        per_request_time = inference_time_ms / batch_size
        total_tokens = sum(r.tokens_generated for r in results)
        tokens_per_sec = (total_tokens / (inference_time_ms / 1000)) if inference_time_ms > 0 else 0

        logger.info(
            f"Batch complete: {batch_size} prompts | "
            f"{inference_time_ms:.0f}ms total | "
            f"{per_request_time:.0f}ms/request | "
            f"{total_tokens} tokens | "
            f"{tokens_per_sec:.1f} tok/s"
        )

        return results

    def _unbatch_pkv(self, batched_pkv, batch_size: int):
        """Unbatch past_key_values from a single forward pass into a list per sequence."""
        unbatched = []
        for i in range(batch_size):
            seq_pkv = []
            for layer in batched_pkv:
                k = layer[0][i:i+1] # shape (1, H, S, D)
                v = layer[1][i:i+1]
                seq_pkv.append((k, v))
            unbatched.append(tuple(seq_pkv))
        return unbatched

    def _sample(self, logits: torch.Tensor, sequences: list[Sequence]) -> torch.Tensor:
        """Sample next tokens from logits based on temperature."""
        next_tokens = []
        for i, seq in enumerate(sequences):
            if seq.temperature > 0:
                probs = torch.softmax(logits[i] / seq.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = torch.argmax(logits[i], dim=-1, keepdim=True)
            next_tokens.append(token.view(1))
        return torch.cat(next_tokens)

    @torch.no_grad()
    def _generate_batch_step(self, sequences: list[Sequence]) -> None:
        """Run one iteration of continuous batching."""
        if not sequences:
            return

        # Split into prefill and decode
        prefill_seqs = [s for s in sequences if s.past_key_values is None]
        decode_seqs = [s for s in sequences if s.past_key_values is not None]

        # --- PREFILL ---
        if prefill_seqs:
            prompts = [s.prompt for s in prefill_seqs]
            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = self._sample(next_token_logits, prefill_seqs)
            unbatched_pkv = self._unbatch_pkv(outputs.past_key_values, len(prefill_seqs))

            for i, seq in enumerate(prefill_seqs):
                seq.past_key_values = unbatched_pkv[i]
                seq.input_ids = input_ids[i].tolist() + [next_tokens[i].item()]
                seq.attention_mask = attention_mask[i].tolist() + [1]
                seq.tokens_generated = 1
                new_text = self.tokenizer.decode([next_tokens[i].item()])
                seq.generated_text += new_text
                if next_tokens[i].item() == self.tokenizer.eos_token_id or seq.tokens_generated >= seq.max_tokens:
                    seq.is_finished = True

        # --- DECODE ---
        if decode_seqs:
            # Pad past_key_values for batching
            max_seq_len = max(s.past_key_values[0][0].size(2) for s in decode_seqs)
            
            batched_pkv = []
            num_layers = self.model.config.n_layer
            for layer_idx in range(num_layers):
                layer_k = []
                layer_v = []
                for seq in decode_seqs:
                    k = seq.past_key_values[layer_idx][0]
                    v = seq.past_key_values[layer_idx][1]
                    pad_len = max_seq_len - k.size(2)
                    if pad_len > 0:
                        k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                    layer_k.append(k)
                    layer_v.append(v)
                batched_pkv.append((torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)))
            
            # Prepare input_ids and attention_mask
            input_ids_list = []
            attention_mask_list = []
            for seq in decode_seqs:
                input_ids_list.append([seq.input_ids[-1]])
                # Sequence's current attention_mask length is seq_len + 1. We pad to max_seq_len + 1.
                pad_len = max_seq_len - (len(seq.attention_mask) - 1)
                # Pad with 0s on the left
                new_mask = [0]*pad_len + seq.attention_mask
                # Update the sequence's attention_mask (will append 1 later)
                seq.attention_mask = new_mask
                attention_mask_list.append(new_mask + [1])

            input_ids = torch.tensor(input_ids_list, device=self.device)
            attention_mask = torch.tensor(attention_mask_list, device=self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=batched_pkv,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = self._sample(next_token_logits, decode_seqs)
            unbatched_pkv = self._unbatch_pkv(outputs.past_key_values, len(decode_seqs))

            for i, seq in enumerate(decode_seqs):
                seq.past_key_values = unbatched_pkv[i]
                seq.input_ids.append(next_tokens[i].item())
                seq.attention_mask.append(1)
                seq.tokens_generated += 1
                new_text = self.tokenizer.decode([next_tokens[i].item()])
                seq.generated_text += new_text
                if next_tokens[i].item() == self.tokenizer.eos_token_id or seq.tokens_generated >= seq.max_tokens:
                    seq.is_finished = True


    @property
    def cache_stats(self) -> dict:
        """Get KV-cache statistics."""
        return self.kv_cache.stats()

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(model={self.model_name}, "
            f"device={self.device}, cache={self.kv_cache})"
        )
