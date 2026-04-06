from typing import Literal


Difficulty = Literal["easy", "medium", "hard"]
AVAILABLE_DIFFICULTIES: tuple[Difficulty, Difficulty, Difficulty] = (
    "easy",
    "medium",
    "hard",
)


def normalize_difficulty(value: str) -> Difficulty:
    """Normalize user-provided difficulty string."""
    normalized = value.strip().lower()
    if normalized not in AVAILABLE_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty: {value}")
    return normalized  # type: ignore[return-value]


def select_difficulty(preferred: str | None = None) -> Difficulty:
    """Return selected difficulty for current episode."""
    if preferred is None:
        return "easy"
    return normalize_difficulty(preferred)
