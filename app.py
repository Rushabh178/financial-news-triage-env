from fastapi import FastAPI

from env import Action, FinancialNewsEnvironment, Observation, Reward, State


app = FastAPI()
ENV = FinancialNewsEnvironment()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    return ENV.reset()


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
