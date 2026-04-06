from .models import Action


FIELD_WEIGHTS = {
    "relevance": 0.25,
    "sector": 0.25,
    "urgency": 0.25,
    "action": 0.25,
}


def grade_action(action: Action, truth: dict[str, str]) -> float:
    """Grade an action against truth using exact literal equality."""
    score = 0.0

    for field, weight in FIELD_WEIGHTS.items():
        if getattr(action, field) == truth.get(field):
            score += weight

    return min(max(score, 0.0), 1.0)
