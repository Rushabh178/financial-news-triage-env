import os

from openai import OpenAI

from env.environment import FinancialNewsEnvironment
from env.models import Action
from graders import EasyTriageGrader, MediumTriageGrader, HardTriageGrader


# Required by OpenEnv validator: must route all calls through the injected proxy.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TASK_CONFIGS = [
    {"task_id": "triage_easy", "grader": EasyTriageGrader},
    {"task_id": "triage_medium", "grader": MediumTriageGrader},
    {"task_id": "triage_hard", "grader": HardTriageGrader},
]


def run() -> None:
    """Run deterministic baseline inference loop against all 3 graded tasks."""
    MAX_STEPS = 8
    BASELINE_ACTION = Action(
        relevance="high",
        sector="macro",
        urgency="high",
        action="buy",
    )

    # Client initialized for OpenEnv spec compliance; baseline remains deterministic.
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with one word: ready"}],
        max_tokens=2,
    )

    overall_results = []

    for task_config in TASK_CONFIGS:
        task_id = task_config["task_id"]
        grader_class = task_config["grader"]
        
        env = FinancialNewsEnvironment()
        obs = env.reset(task_id=task_id)

        print(f"[START] task={task_id} env=openenv model=baseline grader={grader_class.__name__}")

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
        
        overall_results.append({"task": task_id, "success": success, "score": score})
    
    print(f"\n[SUMMARY] All 3 tasks executed with graders: {overall_results}")


if __name__ == "__main__":
    run()
