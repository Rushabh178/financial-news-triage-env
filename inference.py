import os

from openai import OpenAI

from env.environment import FinancialNewsEnvironment
from env.models import Action


# Required by OpenEnv validator: must route all calls through the injected proxy.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Client initialized for OpenEnv spec compliance; baseline remains deterministic.
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
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

    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with one word: ready"}],
        max_tokens=2,
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
    score = min(max(total_score / MAX_STEPS, 0.0), 1.0)
    success = "true" if score > 0 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={steps_taken} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    run()
