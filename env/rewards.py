from typing import Callable


PenaltyHook = Callable[[float], float]


def apply_reward_hooks(base_reward: float, hooks: list[PenaltyHook] | None = None) -> float:
    """Apply reward or penalty hooks to a base reward value."""
    raise NotImplementedError("apply_reward_hooks is not implemented yet.")


def default_penalty_hooks() -> list[PenaltyHook]:
    """Return default penalty hooks.

    Placeholder for future reward shaping logic.
    """
    return []
