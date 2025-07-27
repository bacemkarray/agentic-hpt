from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
import operator

class Parameters(TypedDict):
    num_layers: int
    learning_rate: float
    hidden_dim: int
    dropout: float

class TuningState(TypedDict):
    status: str
    params: Parameters
    score: float
    workers_done: Annotated[List, operator.add]
    iteration: int

class ActionOutput(BaseModel):
    action: str = Field(description="The next action to take")
    reason: str = Field(description="Explanation for the action")