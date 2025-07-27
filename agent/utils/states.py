from typing import TypedDict, Annotated, List
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