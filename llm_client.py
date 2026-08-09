"""Groq LLM client wrapper with retry/backoff for rate limits."""
import time

from groq import Groq, RateLimitError, APIError

import config


class LLMClient:
    def __init__(self, model: str | None = None):
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = model or config.GROQ_MODEL

    def chat(self, system: str, user: str, temperature: float = config.TEMPERATURE) -> str:
        """Single chat completion with retry on rate limit / transient errors."""
        last_error: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    top_p=config.TOP_P,
                    max_tokens=config.MAX_TOKENS,
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                last_error = e
            except APIError as e:
                last_error = e
            except Exception as e:  # network errors etc.
                last_error = e
            delay = config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} attempts: {last_error}")
