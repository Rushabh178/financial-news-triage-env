import os

from openai import OpenAI

from env.environment import FinancialNewsEnvironment
from env.models import Action


# Reserved for future OpenAI / HF router integration
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Client initialized for OpenEnv spec compliance; baseline remains deterministic.
client = None
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )


def run() -> None:
    """Run deterministic baseline inference loop against the environment."""
    MAX_STEPS = 8
    BASELINE_ACTION = Action(
        relevance="high",
        sector="macro",
        urgency="high",
        action="buy",
    )

    env = FinancialNewsEnvironment()
    obs = env.reset()

    print("[START] task=financial_news env=openenv model=baseline")

    rewards: list[float] = []
    steps_taken = 0

    for n in range(1, MAX_STEPS + 1):
        reward = env.step(BASELINE_ACTION)
        rewards.append(reward.reward)
        steps_taken = n
        print(
            f"[STEP] step={n} action={BASELINE_ACTION.action}"
            f" reward={reward.reward:.2f} done={str(reward.done).lower()} error=null"
        )
        if reward.done:
            break

    total_score = sum(rewards)
    success = "true" if total_score > 0 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={steps_taken} rewards={rewards_str}")


if __name__ == "__main__":
    run()
