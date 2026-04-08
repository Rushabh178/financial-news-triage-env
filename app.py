from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import Action, FinancialNewsEnvironment, Observation, Reward, State


app = FastAPI()
ENV = FinancialNewsEnvironment()


class ResetRequest(BaseModel):
    """Reset request payload with optional task_id."""
    task_id: Optional[str] = None


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
