import pytest

from env.environment import FinancialNewsEnvironment


def test_environment_reset_is_stubbed() -> None:
    env = FinancialNewsEnvironment()
    with pytest.raises(NotImplementedError):
        env.reset()


def test_environment_state_is_stubbed() -> None:
    env = FinancialNewsEnvironment()
    with pytest.raises(NotImplementedError):
        env.state()
