from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from pydantic import BaseModel, Field
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


class State(TypedDict):
    status: str
    best_params: Optional[Dict[str, Any]]
    #     "max_depth": 6,
    #     "learning_rate": 0.1,
    #     "n_estimators": 100,
    #     "subsample": 1.0
    # })
    best_score: Optional[float] = 0.0
    workers_done: Annotated[list, operator.add]
    iteration: int = 0

# Global Configurations
DATA_PATH = "src/ml/data/diabetes_prediction_dataset.csv"

# Load Dataset
df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, columns=['smoking_history', 'gender'])
X = df.drop(columns=["diabetes"])
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def make_objective(fixed_params: dict, param_to_tune: str):
    def objective(trial):
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
    def worker(state: State, config: RunnableConfig):
        return worker_tune(state, config, param_name)
    return worker


def worker_tune(state: State, config: RunnableConfig, param_name: str) -> dict:
    conf  = config["configurable"]
    if not conf.get(f"tune_{param_name}", False):
        # skip tuning this param
        return {
            "best_params": state.get("best_params"),
            "best_score": state.get("best_score"),
            "workers_done": state.get("workers_done"),
        }


    # If params don't already exist set default params
    best_params = state.get("best_params", {})
    objective = make_objective(best_params, param_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Update best params with this worker's best value
    new_best_params = best_params.copy()
    new_best_params[param_name] = study.best_params[param_name]
    best_score = max(state.get("best_score", 0.0), study.best_value)

    # Track which workers finished
    workers_done = state.get("workers_done").copy()
    if param_name not in workers_done:
        workers_done.append(param_name)

    return {
        "best_params": new_best_params,
        "best_score": best_score,
        "workers_done": workers_done,
    }


def coordinator(state: State) -> Command:
    required_workers = {"max_depth", "learning_rate", "n_estimators", "subsample"}
    workers_done = set(state.get("workers_done", []))
    best_params = state.get("best_params", {})
    best_score = state.get("best_score", 0.0)
    iteration = state.get("iteration", 0)

    # Wait until all workers finish
    if not required_workers.issubset(workers_done):
        # Still waiting for workers
        return Command(update={}, goto="wait")

    # Aggregate: pick best score and params from workers
    # (Assuming workers update best_params and best_score in state)
    # For simplicity, keep best_score and best_params from last worker (or implement logic here)

    # Reset workers_done for next iteration
    workers_done.clear()
    iteration += 1

    # Stopping condition (e.g., max 3 iterations)
    if iteration >= 1:
        return Command(update={"status": "finalize", "iteration": iteration, "workers_done": workers_done, "best_params": best_params, "best_score": best_score}, 
                       goto="finalize")

    # Continue tuning: send to all workers again
    return Command(update={"status": "tuning", "iteration": iteration, "workers_done": list(workers_done), "best_params": best_params, "best_score": best_score}, 
                   goto="start_workers")


# Do one last model run
def finalize(state: State) -> dict:
    best_params = state.get("best_params", {})
    model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Final model accuracy: {accuracy}")
    return {"status": "__end__", "final_accuracy": accuracy}


def wait(state: State) -> dict:
    # No-op, just a placeholder to wait for workers
    return {}


graph = (
    StateGraph(State)
    .add_node("start_workers", lambda s, c=None: {"status": "tuning"})
    .add_node("tune_max_depth", make_worker("max_depth"))
    .add_node("tune_learning_rate", make_worker("learning_rate"))
    .add_node("tune_n_estimators", make_worker("n_estimators"))
    .add_node("tune_subsample", make_worker("subsample"))
    .add_node("coordinator", coordinator)
    .add_node("finalize", finalize)
    .add_node("wait", wait)

    # Edges
    .add_edge("__start__", "start_workers")
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