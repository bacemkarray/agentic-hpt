from typing import TypedDict, Annotated, List
import operator

class Parameters(TypedDict):
    max_depth: int
    learning_rate: float
    tune_n_estimators: int
    subsample: float

class TuningState(TypedDict):
    status: str
    params: Parameters
    score: float
    workers_done: Annotated[List, operator.add]
    iteration: int