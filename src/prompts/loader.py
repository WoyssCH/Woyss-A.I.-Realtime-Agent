from __future__ import annotations

from pathlib import Path


def load_prompt(filename: str) -> str:
    """Load a prompt text file shipped with the codebase."""

    prompt_dir = Path(__file__).resolve().parent
    path = prompt_dir / filename
    if not path.exists():
        raise RuntimeError(f"Prompt file not found: {filename}")
    return path.read_text(encoding="utf-8").strip() + "\n"
