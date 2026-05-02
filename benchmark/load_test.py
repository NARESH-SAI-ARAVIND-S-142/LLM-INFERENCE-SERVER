"""
miniServe — Locust Load Test
Stress-test the inference server under various concurrent loads.

Usage:
    # Start Locust web UI (opens at http://localhost:8089):
    cd llm-inference-server
    source venv/bin/activate
    locust -f benchmark/load_test.py --host http://localhost:8000
    
    # Headless mode (CLI only):
    locust -f benchmark/load_test.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 60s --csv=benchmark/results/locust
"""

import random
from locust import HttpUser, task, between, events


SAMPLE_PROMPTS = [
    "Once upon a time in a magical forest, there lived",
    "The future of artificial intelligence is shaping up to be",
    "In the year 2050, humanity discovered a way to",
    "The secret to happiness lies in understanding that",
    "Deep beneath the ocean, scientists found evidence of",
    "The last star in the universe flickered and then",
    "A robot walked into a coffee shop and ordered",
    "The ancient library contained books that could predict",
    "On Mars, the first colony celebrated its tenth anniversary",
    "The quantum computer finally solved the problem that",
    "Far beyond the edge of the galaxy, a signal",
    "The invention of teleportation changed society because",
    "In a world where dreams are shared, people began to",
    "The first contact with alien life happened when",
    "Climate change had transformed the landscape into",
]


class InferenceUser(HttpUser):
    """
    Simulates a user sending text generation requests.
    
    Each user sends requests with random prompts from the sample list.
    The wait_time simulates realistic inter-request delays.
    """
    
    # Wait 0.5 to 2 seconds between requests per user
    wait_time = between(0.5, 2.0)

    @task(weight=10)
    def generate_text(self):
        """Send a generation request with a random prompt."""
        prompt = random.choice(SAMPLE_PROMPTS)
        with self.client.post(
            "/v1/generate",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.8,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("tokens_generated", 0)
                latency = data.get("latency_ms", 0)
                batch_size = data.get("batch_size", 1)
                
                # Log custom metrics
                response.success()
            else:
                response.failure(f"Status {response.status_code}: {response.text}")

    @task(weight=1)
    def health_check(self):
        """Periodically check server health."""
        self.client.get("/health")

    @task(weight=1)
    def check_stats(self):
        """Periodically check server stats."""
        self.client.get("/stats")
