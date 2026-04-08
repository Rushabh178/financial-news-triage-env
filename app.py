from typing import Optional

from fastapi import FastAPI, Body
from pydantic import BaseModel

from env import Action, FinancialNewsEnvironment, Observation, Reward, State


app = FastAPI()
ENV = FinancialNewsEnvironment()


class ResetRequest(BaseModel):
    """Reset request payload with optional task_id."""
    task_id: Optional[str] = None


class GraderRequest(BaseModel):
    """Grader invocation request."""
    task_id: str
    action: dict
    truth: dict


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = None) -> Observation:
    task_id = request.task_id if request else None
    return ENV.reset(task_id=task_id)


@app.post("/step", response_model=Reward)
def step(action: Action) -> Reward:
    try:
        return ENV.step(action)
    except RuntimeError:
        ENV.reset()
        return ENV.step(action)


@app.get("/state", response_model=State)
def state() -> State:
    try:
        return ENV.state()
    except RuntimeError:
        ENV.reset()
        return ENV.state()


@app.get("/tasks")
def get_tasks():
    """List available tasks with their grader information."""
    return {
        "tasks": [
            {
                "id": "triage_easy",
                "name": "Easy Financial Triage",
                "difficulty": "easy",
                "grader": "graders.grader_easy:EasyTriageGrader",
                "max_steps": 8,
            },
            {
                "id": "triage_medium",
                "name": "Medium Financial Triage",
                "difficulty": "medium",
                "grader": "graders.grader_medium:MediumTriageGrader",
                "max_steps": 8,
            },
            {
                "id": "triage_hard",
                "name": "Hard Financial Triage",
                "difficulty": "hard",
                "grader": "graders.grader_hard:HardTriageGrader",
                "max_steps": 8,
            },
        ]
    }


@app.post("/grader")
def invoke_grader(request: GraderRequest):
    """Invoke grader for a specific task."""
    from graders import EasyTriageGrader, MediumTriageGrader, HardTriageGrader
    
    graders_map = {
        "triage_easy": EasyTriageGrader,
        "triage_medium": MediumTriageGrader,
        "triage_hard": HardTriageGrader,
    }
    
    if request.task_id not in graders_map:
        return {"error": f"Unknown task: {request.task_id}"}
    
    # Convert dict to Action
    action_obj = Action(**request.action)
    grader_class = graders_map[request.task_id]
    reward = grader_class.grade(action_obj, request.truth)
    return {"reward": reward, "task_id": request.task_id}
