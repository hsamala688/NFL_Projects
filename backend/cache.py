import json
from pathlib import Path
from typing import Any

_CACHE_PATH = Path(__file__).parent / "demo_cache.json"

_cache: dict[str, Any] = {}

if _CACHE_PATH.exists():
    _cache = json.loads(_CACHE_PATH.read_text())


def questions() -> list[str]:
    return list(_cache.keys())


def lookup(question: str) -> dict | None:
    return _cache.get(question)
