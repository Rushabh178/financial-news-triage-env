from __future__ import annotations

from env.grader import grade_action
from env.models import Action


class MediumTriageGrader:
    """Grader used for medium financial triage tasks."""

    @staticmethod
    def grade(action: Action, truth: dict[str, str]) -> float:
        return grade_action(action, truth)
