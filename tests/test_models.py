from env.models import Action, Observation, Reward, State


def test_models_are_importable() -> None:
    assert Action is not None
    assert Observation is not None
    assert Reward is not None
    assert State is not None
