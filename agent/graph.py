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
import mlflow


# TEMP ML STUFF
DATA_PATH = "ml/data/diabetes_prediction_dataset.csv"

# Load Dataset
df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, columns=['smoking_history', 'gender'])
X = df.drop(columns=["diabetes"])
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Create experiment id for mlflow
def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
    else:
      return mlflow.create_experiment(experiment_name)
    




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
    experiment_id: str


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
        "iteration": 0,
        "experiment-id": get_or_create_experiment("Agentic-HPT-Testing") 
    }

def start_workers(state: TuningState) -> TuningState:
    # Just pass state through without resetting anything
    return state

def make_objective(fixed_params: Dict, param_to_tune: str):
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
        acc = accuracy_score(y_test, preds)

        #mlflow logging
        with mlflow.start_run(nested=True):
            mlflow.log_param(param_to_tune, params[param_to_tune])
            mlflow.log_metrics({"accuracy": acc})
            mlflow.set_tag("tuned_param", param_to_tune)
            # Optionally log all fixed params for traceability
            for k, v in fixed_params.items():
                mlflow.log_param(f"fixed_{k}", v)
        return acc

    return objective


def make_worker(param_name: str):

    def worker(state: TuningState):
        return worker_tune(state, param_name)
    return worker


def worker_tune(state: TuningState, param_name: str):
    params = state["params"]

    # Create study
    objective = make_objective(params, param_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    best_val = study.best_params.get(param_name)
    return {
        "workers_done": state["workers_done"] + [param_name],
        "best_param_value": best_val,
        "best_score": study.best_value,
    }


def coordinator(state: TuningState):
    required_workers = {"max_depth", "learning_rate", "n_estimators", "subsample"}
    workers_done = state["workers_done"]

    # Wait until all workers finish
    if not required_workers.issubset(set(workers_done)):
        # Still waiting for workers
        return Command(update={}, goto="wait")

    iteration = state["iteration"] + 1
    
    experiment_id = state["experiment_id"]  # You should store this in your state
    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")
    
    # Group by tuned_param and find the best run in each group
    best_per_param = (
        runs.groupby("tags.tuned_param")
        .apply(lambda df: df.loc[df["metrics.accuracy"].idxmax()])
    )

    # Find which parameter's best run had the highest accuracy
    best_overall = best_per_param.loc[best_per_param["metrics.accuracy"].idxmax()]
    best_param = best_overall["tags.tuned_param"]

    # Extract the best value for the parameter (params.<param_name>)
    best_val = best_overall[f"params.{best_param}"]
    best_score = best_overall["metrics.accuracy"]

    # Build the new global params & score
    new_params = {**state["params"], best_param: best_val}
    new_state = {
        "params": new_params,
        "score": best_score,
        "workers_done": [],
        "iteration": iteration
    }

    # Stopping condition (will let LLM decide this in the future)
    if iteration > 2:
        return Command(update={**new_state, "status": "finalize"},
                       goto="finalize")
    
    
    # Continue tuning: send to all workers again
    return Command(update={**new_state},
                    goto="start_workers")


# Do one last model run
def finalize(state: TuningState):
    params = state["params"]
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Final model accuracy: {accuracy}")
    return {}


def wait(state: TuningState):
    return Command(goto="coordinator")

from langgraph.graph import START, END


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
    .add_edge(START, "define_parameters")
    .add_edge("define_parameters", "start_workers")
    .add_edge("start_workers", "tune_max_depth")
    .add_edge("start_workers", "tune_learning_rate")
    .add_edge("start_workers", "tune_n_estimators")
    .add_edge("start_workers", "tune_subsample")
    .add_edge("tune_max_depth", "coordinator")
    .add_edge("tune_learning_rate", "coordinator")
    .add_edge("tune_n_estimators", "coordinator")
    .add_edge("tune_subsample", "coordinator")
    # .add_edge("coordinator", "wait")
    .add_edge("wait", "coordinator")
    .add_edge("finalize", END)
    .compile(name="Parallel HP Tuning Graph")
)


initial_input = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
}

result = graph.invoke(initial_input)