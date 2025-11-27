"""Utility functions for caption text preprocessing."""

from __future__ import annotations

import re
from typing import List

_CONTROL_CHAR_PATTERN = re.compile(r"[\u0000-\u001F\u007F]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?！？。]+")


def clean_caption(text: str, lowercase: bool = True) -> str:
    """Return a normalized caption string."""
    if not isinstance(text, str):
        return ""

    normalized = _CONTROL_CHAR_PATTERN.sub(" ", text).strip()
    if lowercase:
        normalized = normalized.lower()
    normalized = _MULTI_SPACE_PATTERN.sub(" ", normalized)
    return normalized


def split_sentences(text: str) -> List[str]:
    """Split a caption into sentences using a lightweight regex."""
    if not text:
        return []

    sentences = []
    for candidate in _SENTENCE_SPLIT_PATTERN.split(text):
        candidate = candidate.strip()
        if candidate:
            sentences.append(candidate)
    return sentences


__all__ = ["clean_caption", "split_sentences"]

