from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Relevance = Literal["low", "medium", "high"]
Urgency = Literal["low", "medium", "high"]
ActionType = Literal["buy", "hold", "sell", "ignore"]

Sector = Literal[
    "technology",
    "finance",
    "energy",
    "healthcare",
    "industrials",
    "consumer",
    "real_estate",
    "materials",
    "utilities",
    "telecom",
    "macro",
    "other",
]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    relevance: Relevance
    sector: Sector
    urgency: Urgency
    action: ActionType


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    headline: str = Field(min_length=1)
    index: int = Field(ge=0)
    total: int = Field(ge=1)


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reward: float = Field(ge=0.0, le=1.0)
    done: bool


class State(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    observation: Observation
    score: float = Field(ge=0.0)
    steps_taken: int = Field(ge=0)
    done: bool
