"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

import os
import mlflow
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt

# Unneeded for now
class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


class State(TypedDict):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    
    status: str  # Workflow status (e.g., "monitoring", "retraining")

async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    return {
        "changeme": "output from call_model. "
        f'Configured with {configuration.get("my_configurable_param")}'
    }

# Global Configurations
MODEL_VERSION = "v1.0"
MODEL_ACCURACY = 0.0
EXPERIMENT_NAME = "MLOps_Agent"
DATA_PATH = "src/ml/data/diabetes_prediction_dataset.csv"

# Load Dataset
df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, columns=['smoking_history', 'gender'])
X = df.drop(columns=["diabetes"])
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# Hyperparameter Optimization Function
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


# Train Initial Model
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_params = study.best_params
model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
MODEL_ACCURACY = accuracy_score(y_test, model.predict(X_test))


# Log Model to MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", MODEL_ACCURACY)
    mlflow.xgboost.log_model(model, "model")

from langgraph.types import Command

# MLOps Workflow Functions
def model_perf(state: State) -> Command[Literal["decide", "__end__"]]:
    """Monitor model performance."""
    global MODEL_ACCURACY
    print(f"Monitoring Model: {MODEL_VERSION}, Accuracy: {MODEL_ACCURACY}")

    value = "drift_detected" if MODEL_ACCURACY < 0.75 else "model_healthy"
    goto = "decide_retraining" if value == "drift_detected" else "__end__"
    return Command(
        update={"status": value},
        goto=goto
    )

def decide_retrain(state: State) -> Command[Literal["retrain", "__end__"]]:
    """Decide if retraining is needed using an LLM."""
    response = llm.predict("The model accuracy dropped below the threshold. Should I retrain?")
    value = "retrain" if "yes" in response.lower() else "__end__"
    goto = "deploy" if value == "retrain" else "__end__"
    return Command(
        update={"status": value},
        goto=goto
    )

def retrain_model(state):
    """Retrain model with updated dataset & hyperparameters."""
    global MODEL_ACCURACY, MODEL_VERSION
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    new_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
    new_model.fit(X_train, y_train)
    MODEL_ACCURACY = accuracy_score(y_test, new_model.predict(X_test))
    MODEL_VERSION = "v" + str(int(MODEL_VERSION.split("v")[-1]) + 1)
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("new_accuracy", MODEL_ACCURACY)
        mlflow.xgboost.log_model(new_model, "model")
    return {"status": "deploy"}

def deploy_model(state):
    """Deploy the updated model."""
    print(f"Deploying Model {MODEL_VERSION} with Accuracy: {MODEL_ACCURACY}")
    return {"status": "__end__"}






# # Define the graph
# graph = (
#     StateGraph(State, config_schema=Configuration)
#     .add_node(call_model)
#     .add_edge("__start__", "call_model")
#     .compile(name="New Graph")
# )

# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node("monitor", model_perf)
    .add_node("decide", decide_retrain)
    .add_node("retrain", retrain_model)
    .add_node("deploy", deploy_model)

    # Logic
    .add_edge("__start__", "monitor")
    .add_edge("retrain", "deploy")
    .add_edge("deploy", "__end__")
    .compile(name="Initial Graph")
)
