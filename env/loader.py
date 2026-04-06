from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tasks import Difficulty, normalize_difficulty


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def difficulty_file_path(difficulty: str) -> Path:
    """Return the resolved JSON path for a difficulty level."""
    normalized: Difficulty = normalize_difficulty(difficulty)
    return DATA_DIR / f"{normalized}.json"


def load_difficulty_data(difficulty: str) -> list[dict[str, Any]]:
    """Safely load and validate JSON records for a difficulty.

    Validation details are intentionally deferred to later milestones.
    """
    file_path = difficulty_file_path(difficulty)
    if not file_path.is_file():
        raise ValueError(f"Difficulty data file does not exist: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Difficulty data JSON root must be a list.")

    if not data:
        raise ValueError("Difficulty data list must not be empty.")

    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Difficulty data items must all be dictionaries.")

    return data
