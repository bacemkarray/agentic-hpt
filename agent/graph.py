from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.types import Command

import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# TEMP ML STUFF
DATA_PATH = "src/ml/data/diabetes_prediction_dataset.csv"

# Load Dataset
df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, columns=['smoking_history', 'gender'])
X = df.drop(columns=["diabetes"])
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class Parameters(TypedDict):
    max_depth: int
    learning_rate: float
    tune_n_estimators: int
    subsample: float

class TuningState(TypedDict):
    status: str
    params: Parameters
    score: float
    workers_done: Annotated[list, operator.add]
    worker_reports: Annotated[dict, operator.add]
    iteration: int


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def initialize_params(state: Parameters) -> TuningState:
    params = {
        "max_depth": state.get("max_depth", 6),
        "learning_rate": state.get("learning_rate", 0.1),
        "n_estimators": state.get("n_estimators", 100),
        "subsample": state.get("subsample", 1.0)
    }
    
    return {
        "status": "tuning",
        "params": params,
        "score": 0.0,
        "workers_done": [],
        "iteration": 0
    }

def start_workers(state: TuningState) -> TuningState:
    # Just pass state through without resetting anything
    return state

def make_objective(fixed_params: dict, param_to_tune: str):
    def objective(trial):
        # Copy to isolate changes between workers
        params = fixed_params.copy()
        if param_to_tune == "max_depth":
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        elif param_to_tune == "learning_rate":
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        elif param_to_tune == "n_estimators":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
        elif param_to_tune == "subsample":
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        else:
            raise ValueError(f"Unknown hyperparameter {param_to_tune}")

        model = xgb.XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    return objective


def make_worker(param_name: str):
    def worker(state: TuningState):
        return worker_tune(state, param_name)
    return worker


def worker_tune(state: TuningState, param_name: str) -> dict:
    params = state["params"]

    # Create study
    objective = make_objective(params, param_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Extract only this worker's parameter from the best_params found in the study
    best_param   = study.best_params[param_name]
    best_score = max(state["score"], study.best_value)
    
    # Keep all params fixed except the one the worker tuned
    new_params = {**params, param_name: best_param}
    # Add worker's completion to state
    new_workers_done = state["workers_done"] + [param_name]

    return {
        "params": new_params,
        "score": best_score,
        "workers_done": new_workers_done
    }


def coordinator(state: TuningState) -> Command:
    required_workers = {"max_depth", "learning_rate", "n_estimators", "subsample"}
    workers_done = state["workers_done"]
    params = state["params"]
    score = state["score"]
    iteration = state["iteration"] + 1

    # Wait until all workers finish
    if not required_workers.issubset(set(workers_done)):
        # Still waiting for workers
        return Command(update={}, goto="wait")

    # Aggregate: pick best score and params from workers
    # (Assuming workers update params and score in state)
    # For simplicity, keep score and params from last worker (or implement logic here)

    # Reset workers_done for next iteration

    # Stopping condition
    if iteration >= 1:
        return Command(update={"status": "finalize", 
                               "iteration": iteration, 
                               "workers_done": workers_done, 
                               "params": params, 
                               "score": score}, 
                       goto="finalize")
    
    workers_done = []

    # Continue tuning: send to all workers again
    return Command(update={"status": "tuning", 
                           "iteration": iteration, 
                           "workers_done": workers_done, 
                           "params": params, 
                           "score": score}, 
                   goto="start_workers")


# Do one last model run
def finalize(state: TuningState) -> dict:
    params = state["params"]
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Final model accuracy: {accuracy}")
    return {"status": "__end__", "final_accuracy": accuracy}


def wait(state: TuningState) -> dict:
    # No-op, just a placeholder to wait for workers
    return {}


graph = (
    StateGraph(TuningState)
    # Nodes
    .add_node("define_parameters", initialize_params)
    .add_node("start_workers", start_workers)
    .add_node("tune_max_depth", make_worker("max_depth"))
    .add_node("tune_learning_rate", make_worker("learning_rate"))
    .add_node("tune_n_estimators", make_worker("n_estimators"))
    .add_node("tune_subsample", make_worker("subsample"))
    .add_node("coordinator", coordinator)
    .add_node("finalize", finalize)
    .add_node("wait", wait)

    # Edges
    .add_edge("__start__", "define_parameters")
    .add_edge("define_parameters", "start_workers")
    .add_edge("start_workers", "tune_max_depth")
    .add_edge("start_workers", "tune_learning_rate")
    .add_edge("start_workers", "tune_n_estimators")
    .add_edge("start_workers", "tune_subsample")
    .add_edge("tune_max_depth", "coordinator")
    .add_edge("tune_learning_rate", "coordinator")
    .add_edge("tune_n_estimators", "coordinator")
    .add_edge("tune_subsample", "coordinator")
    .add_edge("coordinator", "start_workers")
    .add_edge("coordinator", "finalize")
    .add_edge("finalize", "__end__")
    .add_edge("coordinator", "wait")
    .add_edge("wait", "coordinator")
    .compile(name="Parallel HP Tuning Graph")
)