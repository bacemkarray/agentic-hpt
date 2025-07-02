"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

import mlflow
from mlflow.models.signature import infer_signature
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_openai import ChatOpenAI

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

# Enable MLflow traces
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("auto-tracing-demo")

mlflow.langchain.autolog()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# Hyperparameter Optimization Function
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
    }
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Train Initial Model
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_params = study.best_params
model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
model.fit(X_train, y_train)
MODEL_ACCURACY = accuracy_score(y_test, model.predict(X_test))

signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train.head(3)


# Log Model to MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", MODEL_ACCURACY)
    mlflow.xgboost.log_model(xgb_model=model, name="model", signature=signature, input_example=input_example)

from langgraph.types import Command

# MLOps Workflow Functions
def model_perf(state: State) -> Command[Literal["decide", "__end__"]]:
    """Monitor model performance."""
    global MODEL_ACCURACY
    print(f"Monitoring Model: {MODEL_VERSION}, Accuracy: {MODEL_ACCURACY}")

    value = "drift_detected" if MODEL_ACCURACY < 0.99 else "model_healthy"
    goto = "decide" if value == "drift_detected" else "__end__"
    return Command(
        update={"status": value},
        goto=goto
    )

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class RetrainDecision(BaseModel):
    retrain: bool = Field(description="Whether the model should be retrained")

json_parser = JsonOutputParser(pydantic_object=RetrainDecision)

prompt_template = PromptTemplate(
    template="""
The model accuracy dropped below the threshold.
Should I retrain the model? Please answer with a JSON object like:
{format_instructions}
""",
    input_variables=[],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

def decide_retrain(state: State) -> Command[Literal["retrain", "__end__"]]:
    """Decide if retraining is needed using an LLM."""
    prompt_text = prompt_template.format({}) # don't need to send any input_variables
    response = llm.invoke(prompt_text)
    parsed = json_parser.parse(response)
    retrain_decision = parsed.retrain

    status = "needs_retrain" if retrain_decision else "model_healthy"
    goto = "retrain" if retrain_decision else "__end__"

    return Command(
        update={"status": status},
        goto=goto
    )

def retrain_model(state):
    """Retrain model with updated dataset & hyperparameters."""
    global MODEL_ACCURACY, MODEL_VERSION
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    new_model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
    new_model.fit(X_train, y_train)
    MODEL_ACCURACY = accuracy_score(y_test, new_model.predict(X_test))
    raw = MODEL_VERSION.lstrip("v")           # "1.0"
    major = int(float(raw))                   # float("1.0") → 1.0 → int → 1
    MODEL_VERSION = f"v{major + 1}"           # → "v2"
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("new_accuracy", MODEL_ACCURACY)
        mlflow.xgboost.log_model(xgb_model=new_model, name="model", signature=signature,input_example=input_example)
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
    .add_edge("monitor", "decide")
    .add_edge("monitor", "__end__")
    
    .add_edge("decide", "retrain")
    .add_edge("decide", "__end__")

    .add_edge("retrain", "deploy")
    .add_edge("deploy", "__end__")
    .compile(name="Initial Graph")
)