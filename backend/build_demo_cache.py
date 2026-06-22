"""
Run the agent over a curated question set and write backend/demo_cache.json.
The demo path serves these cached responses with no live LLM call.

Run from repo root:
    python backend/build_demo_cache.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.respond import ask

QUESTIONS = [
    "Who was the most accurate QB in 2025 by model CPOE?",
    "Who led the NFL in EPA per play in 2023?",
    "What is model CPOE and how does it differ from raw completion percentage?",
]

CACHE_PATH = Path(__file__).parent / "demo_cache.json"


def build():
    cache: dict = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    for q in QUESTIONS:
        if q in cache:
            print(f"Skipping (cached): {q}")
            continue
        print(f"Running: {q}")
        result = ask(q)
        cache[q] = result.model_dump()
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        print(f"  answer: {result.answer[:80]}...")
        print()
    print(f"Cache at {CACHE_PATH} ({len(cache)}/{len(QUESTIONS)} questions complete)")


if __name__ == "__main__":
    build()
