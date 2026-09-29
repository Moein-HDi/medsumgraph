"""LLM client using OpenRouter (single provider, no fallback)."""

import time

from openai import OpenAI

import config


class LLMClient:
    def __init__(self, model: str | None = None):
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
                "and fill in your key from https://openrouter.ai/keys."
            )
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )
        self.model = model or config.OPENROUTER_MODEL

    @staticmethod
    def _messages(system: str, user: str):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _call_once(
        self, system: str, user: str, temperature: float, model: str | None = None
    ) -> str:
        chosen_model = model or self.model
        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=self._messages(system, user),
            temperature=temperature,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
            reasoning_effort="none",
            extra_body={"thinking": {"type": "disabled"}},
        )
        # Guard against malformed responses: OpenRouter sometimes returns
        # choices=None or an empty list on 402/closed-account/rate-limit
        # errors that don't raise an exception through the SDK.
        if not response or not getattr(response, "choices", None):
            raise RuntimeError(f"LLM returned no choices: {response}")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"LLM returned None content (finish_reason={getattr(response.choices[0], 'finish_reason', '?')}): "
                f"{response.choices[0]}"
            )
        return content.strip()

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = config.TEMPERATURE,
        model: str | None = None,
    ) -> str:
        """Chat completion via OpenRouter, with retry/backoff on transient errors.

        Pass model= to override the default for this call (used by the KG
        build to use a cheaper model than the QA-time default).
        """
        chosen_model = model or self.model
        last_error: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=chosen_model,
                    messages=self._messages(system, user),
                    temperature=temperature,
                    top_p=config.TOP_P,
                    max_tokens=config.MAX_TOKENS,
                    reasoning_effort="none",
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
            time.sleep(config.LLM_RETRY_BASE_DELAY * (2**attempt))
        raise RuntimeError(
            f"LLM call failed after {config.LLM_MAX_RETRIES} attempts: {last_error}"
        )
