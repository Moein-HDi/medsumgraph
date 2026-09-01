"""LLM client with configurable primary provider (Groq or Liara) + fallback."""

import time

from groq import Groq
from openai import OpenAI

import config


class LLMClient:
    def __init__(self, model: str | None = None):
        if config.LLM_PROVIDER == "liara" and not config.LIARA_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER is 'liara' but LIARA_API_KEY is not set. Fix .env."
            )
        if config.LLM_PROVIDER == "groq" and not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )

        self.model = model or config.GROQ_MODEL
        self.groq = (
            Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT)
            if config.GROQ_API_KEY
            else None
        )
        self.liara = (
            OpenAI(
                base_url=config.LIARA_BASE_URL,
                api_key=config.LIARA_API_KEY,
                timeout=config.LLM_TIMEOUT,
                max_retries=0,
            )
            if config.LIARA_API_KEY
            else None
        )

    @staticmethod
    def _messages(system: str, user: str):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _groq_once(self, system: str, user: str, temperature: float) -> str:
        response = self.groq.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            temperature=temperature,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def _liara_once(self, system: str, user: str, temperature: float) -> str:
        response = self.liara.chat.completions.create(
            model=config.LIARA_MODEL,
            messages=self._messages(system, user),
            temperature=temperature,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
        )

        return response.choices[0].message.content.strip()

    def _run_with_retries(self, call, attempts: int) -> str:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return call()
            except Exception as e:
                last_error = e
            time.sleep(config.LLM_RETRY_BASE_DELAY * (2**attempt))
        raise last_error

    def chat(
        self, system: str, user: str, temperature: float = config.TEMPERATURE
    ) -> str:
        """Chat completion via the configured primary provider, with fallback."""
        if config.LLM_PROVIDER == "groq":
            primary, fallback = self._groq_once, self._liara_once
        else:
            primary, fallback = self._liara_once, self._groq_once

        providers = [p for p in (primary,) if p is not None]
        if not providers:
            raise RuntimeError(
                "No LLM provider configured (both Groq and Liara missing)."
            )

        last_error: Exception | None = None
        tries = config.LLM_MAX_RETRIES
        for i, call in enumerate(providers):
            attempts = tries if i == 0 else min(3, tries)
            try:
                return self._run_with_retries(
                    lambda c=call: c(system, user, temperature), attempts
                )
            except Exception as e:
                last_error = e
                print(last_error)
            tries = min(3, tries)

        raise RuntimeError(
            f"LLM call failed after primary (Groq: {config.LLM_MAX_RETRIES} / "
            f"Liara: {config.LLM_MAX_RETRIES}) attempts: {last_error}"
        )
