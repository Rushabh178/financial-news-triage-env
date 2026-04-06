from __future__ import annotations

from typing import Any

from .grader import grade_action
from .loader import load_difficulty_data
from .models import Action, Observation, Reward, State


class FinancialNewsEnvironment:
    """Stateful Financial News Triage environment."""

    def __init__(self, difficulty: str = "easy") -> None:
        self.difficulty = difficulty
        self._records: list[dict[str, Any]] = []
        self._index = 0
        self._score = 0.0
        self._done = False

    def reset(self) -> Observation:
        """Reset environment and return the initial observation."""
        self._records = load_difficulty_data(self.difficulty)
        self._index = 0
        self._score = 0.0
        self._done = False
        return self._observation_for_index(self._index)

    def state(self) -> State:
        """Return current state."""
        if not self._records:
            raise RuntimeError("Environment must be reset before calling state().")

        observation_index = self._index
        if self._done:
            observation_index = len(self._records) - 1

        return State(
            observation=self._observation_for_index(observation_index),
            score=self._score,
            steps_taken=self._index,
            done=self._done,
        )

    def step(self, action: Action) -> Reward:
        """Apply an action and return reward payload."""
        if self._done:
            return Reward(reward=0.0, done=True)

        if not self._records:
            raise RuntimeError("Environment must be reset before calling step().")

        record = self._records[self._index]
        truth = record.get("truth")
        step_reward = grade_action(action, truth if isinstance(truth, dict) else {})

        self._score += step_reward
        self._index += 1
        self._done = self._index >= len(self._records)

        return Reward(reward=step_reward, done=self._done)

    @staticmethod
    def build_observation(headline: str, index: int, total: int) -> Observation:
        """Create normalized observation model from raw values."""
        return Observation(headline=headline, index=index, total=total)

    def _observation_for_index(self, index: int) -> Observation:
        record = self._records[index]
        return self.build_observation(
            headline=record["headline"],
            index=index,
            total=len(self._records),
        )
