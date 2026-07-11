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
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
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
        self.draft_model_name = config.DRAFT_MODEL
        self.device = device or config.DEVICE
        self.kv_cache = KVCacheManager(max_entries=kv_cache_max or config.KV_CACHE_MAX_ENTRIES)

        logger.info(f"Loading main model '{self.model_name}' on device '{self.device}'...")
        load_start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        # Load draft model
        logger.info(f"Loading draft model '{self.draft_model_name}'...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_name,
            dtype=torch.float32,
        ).to(self.device)
        self.draft_model.eval()

        # CPU Hardware Acceleration
        if self.device == "cpu":
            logger.info("Applying Dynamic Quantization (qint8) to save RAM...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.draft_model = torch.quantization.quantize_dynamic(
                self.draft_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        load_time = time.time() - load_start
        logger.info(
            f"Models loaded in {load_time:.1f}s | "
            f"Device: {self.device} | Speculative K: {config.SPECULATIVE_K}"
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
        """Unbatch past_key_values from a single forward pass into a list per sequence.
        
        Handles both legacy tuple-of-tuples format (GPT-2) and modern DynamicCache (Qwen, Llama).
        Always stores per-sequence KV as a tuple-of-tuples for uniform internal representation.
        """
        # Check if it's a DynamicCache (has key_cache/value_cache attributes)
        if hasattr(batched_pkv, 'key_cache'):
            # DynamicCache format: key_cache[layer] = (batch, heads, seq, dim)
            unbatched = []
            for i in range(batch_size):
                seq_pkv = []
                for layer_idx in range(len(batched_pkv.key_cache)):
                    k = batched_pkv.key_cache[layer_idx][i:i+1]
                    v = batched_pkv.value_cache[layer_idx][i:i+1]
                    seq_pkv.append((k, v))
                unbatched.append(tuple(seq_pkv))
            return unbatched
        else:
            # Legacy tuple-of-tuples format (GPT-2)
            unbatched = []
            for i in range(batch_size):
                seq_pkv = []
                for layer in batched_pkv:
                    k = layer[0][i:i+1]
                    v = layer[1][i:i+1]
                    seq_pkv.append((k, v))
                unbatched.append(tuple(seq_pkv))
            return unbatched

    def _build_cache(self, batched_pkv_tuples):
        """Convert a list of (key, value) tuples into a DynamicCache if available, else return as tuple."""
        try:
            from transformers import DynamicCache
            cache = DynamicCache()
            for layer_k, layer_v in batched_pkv_tuples:
                cache.update(layer_k, layer_v, len(cache))
            return cache
        except ImportError:
            return tuple(batched_pkv_tuples)

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
        """Run one iteration of continuous batching with Speculative Decoding."""
        if not sequences:
            return

        prefill_seqs = [s for s in sequences if s.past_key_values is None]
        decode_seqs = [s for s in sequences if s.past_key_values is not None]

        # --- PREFILL ---
        if prefill_seqs:
            prompts = []
            for s in prefill_seqs:
                prompt_text = ""
                try:
                    if s.messages:
                        prompt_text = self.tokenizer.apply_chat_template(s.messages, tokenize=False, add_generation_prompt=True)
                    elif s.prompt:
                        prompt_text = self.tokenizer.apply_chat_template([{"role": "user", "content": s.prompt}], tokenize=False, add_generation_prompt=True)
                except Exception:
                    if s.messages:
                        prompt_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in s.messages]) + "\nassistant: "
                    else:
                        prompt_text = s.prompt or ""
                prompts.append(prompt_text)

            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Main model prefill
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = self._sample(next_token_logits, prefill_seqs)
            unbatched_pkv = self._unbatch_pkv(outputs.past_key_values, len(prefill_seqs))

            # Draft model prefill
            draft_outputs = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            draft_unbatched_pkv = self._unbatch_pkv(draft_outputs.past_key_values, len(prefill_seqs))

            for i, seq in enumerate(prefill_seqs):
                seq.past_key_values = unbatched_pkv[i]
                seq.draft_past_key_values = draft_unbatched_pkv[i]
                seq.input_ids = input_ids[i].tolist() + [next_tokens[i].item()]
                seq.attention_mask = attention_mask[i].tolist() + [1]
                seq.tokens_generated = 1
                
                new_text = self.tokenizer.decode([next_tokens[i].item()], skip_special_tokens=True)
                seq.latest_token_text = new_text
                seq.generated_text += new_text
                
                if next_tokens[i].item() == self.tokenizer.eos_token_id or seq.tokens_generated >= seq.max_tokens:
                    seq.is_finished = True

        # --- DECODE (SPECULATIVE) ---
        if decode_seqs:
            K = config.SPECULATIVE_K

            # 1. Drafting Phase
            draft_tokens_per_seq = [[] for _ in decode_seqs]
            draft_pkv_per_seq = [seq.draft_past_key_values for seq in decode_seqs]
            current_draft_input_ids = [[seq.input_ids[-1]] for seq in decode_seqs]
            base_masks = [list(seq.attention_mask) for seq in decode_seqs]

            draft_num_layers = getattr(self.draft_model.config, 'n_layer', None) or getattr(self.draft_model.config, 'num_hidden_layers', None)

            for k in range(K):
                max_pkv_len = max(pkv[0][0].size(2) for pkv in draft_pkv_per_seq)
                
                batched_draft_pkv_tuples = []
                for layer_idx in range(draft_num_layers):
                    layer_k = []
                    layer_v = []
                    for i, pkv in enumerate(draft_pkv_per_seq):
                        key = pkv[layer_idx][0]
                        val = pkv[layer_idx][1]
                        pad_len = max_pkv_len - key.size(2)
                        if pad_len > 0:
                            key = torch.nn.functional.pad(key, (0, 0, pad_len, 0))
                            val = torch.nn.functional.pad(val, (0, 0, pad_len, 0))
                        layer_k.append(key)
                        layer_v.append(val)
                    batched_draft_pkv_tuples.append((torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)))
                
                past_key_values = self._build_cache(batched_draft_pkv_tuples)

                attention_mask_list = []
                for i in range(len(decode_seqs)):
                    actual_pkv_len = draft_pkv_per_seq[i][0][0].size(2)
                    pad_len = max_pkv_len - actual_pkv_len
                    mask = [0]*pad_len + base_masks[i] + [1]
                    attention_mask_list.append(mask)
                    base_masks[i].append(1)
                
                draft_input_ids = torch.tensor(current_draft_input_ids, device=self.device)
                draft_attention_mask = torch.tensor(attention_mask_list, device=self.device)

                draft_outputs = self.draft_model(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                next_token_logits = draft_outputs.logits[:, -1, :]
                next_tokens = self._sample(next_token_logits, decode_seqs)
                draft_pkv_per_seq = self._unbatch_pkv(draft_outputs.past_key_values, len(decode_seqs))

                current_draft_input_ids = []
                for i in range(len(decode_seqs)):
                    tok = next_tokens[i].item()
                    draft_tokens_per_seq[i].append(tok)
                    current_draft_input_ids.append([tok])

            # 2. Main Model Verification
            main_pkv_per_seq = [seq.past_key_values for seq in decode_seqs]
            max_main_pkv_len = max(pkv[0][0].size(2) for pkv in main_pkv_per_seq)
            main_num_layers = getattr(self.model.config, 'n_layer', None) or getattr(self.model.config, 'num_hidden_layers', None)

            batched_main_pkv_tuples = []
            for layer_idx in range(main_num_layers):
                layer_k = []
                layer_v = []
                for i, pkv in enumerate(main_pkv_per_seq):
                    key = pkv[layer_idx][0]
                    val = pkv[layer_idx][1]
                    pad_len = max_main_pkv_len - key.size(2)
                    if pad_len > 0:
                        key = torch.nn.functional.pad(key, (0, 0, pad_len, 0))
                        val = torch.nn.functional.pad(val, (0, 0, pad_len, 0))
                    layer_k.append(key)
                    layer_v.append(val)
                batched_main_pkv_tuples.append((torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)))
            
            main_past_key_values = self._build_cache(batched_main_pkv_tuples)

            main_input_ids_list = []
            main_attention_mask_list = []
            for i, seq in enumerate(decode_seqs):
                inputs = [seq.input_ids[-1]] + draft_tokens_per_seq[i][:-1]
                main_input_ids_list.append(inputs)

                actual_pkv_len = main_pkv_per_seq[i][0][0].size(2)
                pad_len = max_main_pkv_len - actual_pkv_len
                mask = [0]*pad_len + seq.attention_mask + [1]*(K-1)
                main_attention_mask_list.append(mask)

            main_input_ids = torch.tensor(main_input_ids_list, device=self.device)
            main_attention_mask = torch.tensor(main_attention_mask_list, device=self.device)

            main_outputs = self.model(
                input_ids=main_input_ids,
                attention_mask=main_attention_mask,
                past_key_values=main_past_key_values,
                use_cache=True,
            )
            
            main_pkv_per_seq = self._unbatch_pkv(main_outputs.past_key_values, len(decode_seqs))
            
            # Batched sampling for K tokens
            next_tokens_main = []
            for i, seq in enumerate(decode_seqs):
                seq_logits = main_outputs.logits[i] # (K, vocab)
                if seq.temperature > 0:
                    probs = torch.softmax(seq_logits / seq.temperature, dim=-1)
                    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (K,)
                else:
                    tokens = torch.argmax(seq_logits, dim=-1) # (K,)
                next_tokens_main.append(tokens.tolist())
            
            # 3. Acceptance & Rollback
            import server.metrics as metrics
            for i, seq in enumerate(decode_seqs):
                draft_toks = draft_tokens_per_seq[i]
                main_toks = next_tokens_main[i]
                
                n = 0
                for j in range(K-1):
                    if draft_toks[j] == main_toks[j]:
                        n += 1
                    else:
                        break
                
                accepted_tokens = draft_toks[:n] + [main_toks[n]]
                
                if K > 1:
                    metrics.SPECULATIVE_TOKENS_GENERATED.inc(K-1)
                    metrics.SPECULATIVE_TOKENS_ACCEPTED.inc(n)
                
                # Rollback KV cache
                L = seq.past_key_values[0][0].size(2)
                target_len = L + 1 + n
                
                def slice_pkv(pkv, target_len):
                    sliced = []
                    for k_t, v_t in pkv:
                        sliced.append((k_t[:, :, :target_len, :], v_t[:, :, :target_len, :]))
                    return tuple(sliced)
                
                seq.draft_past_key_values = slice_pkv(draft_pkv_per_seq[i], target_len)
                seq.past_key_values = slice_pkv(main_pkv_per_seq[i], target_len)
                
                seq.input_ids.extend(accepted_tokens)
                seq.attention_mask.extend([1] * len(accepted_tokens))
                seq.tokens_generated += len(accepted_tokens)
                
                generated_token_ids = seq.input_ids[-seq.tokens_generated:]
                new_total_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                
                seq.latest_token_text = new_total_text[len(seq.generated_text):]
                seq.generated_text = new_total_text
                
                if self.tokenizer.eos_token_id in accepted_tokens or seq.tokens_generated >= seq.max_tokens:
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
