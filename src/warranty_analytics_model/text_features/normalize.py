"""Target-independent Unicode normalization and technical tokenization."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Literal, cast

from .models import TextFeatureSettings

_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_TOKEN = re.compile(r"(?u)[^\W_]+")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(value is not None and value is not False and value != value)
    except (TypeError, ValueError):
        return False


def normalize_description(value: Any, settings: TextFeatureSettings | None = None) -> str | None:
    """Normalize one description; null/blank values become None, never literal text."""

    policy = settings or TextFeatureSettings()
    if _is_missing(value):
        return None
    text = unicodedata.normalize(
        cast(Literal["NFC", "NFD", "NFKC", "NFKD"], policy.unicode_form), str(value)
    )
    if policy.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    if policy.trim:
        text = text.strip()
    if policy.lowercase:
        text = text.casefold()
    return text or None


def technical_tokens(text: str | None) -> list[str]:
    """Return Unicode word-like sequences containing letters or numbers."""

    return _TOKEN.findall(text or "")
