"""Wikipedia summary retrieval (paper uses LangChain's WikipediaRetriever)."""
import wikipediaapi

import config

_wiki = None


def _get_wiki():
    global _wiki
    if _wiki is None:
        # 'en' user agent; lang is English as in the paper
        _wiki = wikipediaapi.Wikipedia("MedSumGraph/1.0", "en")
    return _wiki


def fetch_summary(title: str, max_words: int = config.WIKIPEDIA_MAX_WORDS) -> str:
    """Return the summary paragraph(s) for a Wikipedia page, capped in length."""
    page = _get_wiki().page(title)
    if not page.exists():
        return ""
    text = page.summary
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "..."
    return text
