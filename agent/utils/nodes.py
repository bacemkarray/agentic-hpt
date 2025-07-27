from agent.utils.states import Parameters, TuningState, ActionOutput
from ml.mlp_core import train_and_eval
from typing import Dict
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import optuna
import mlflow
import torch
from pathlib import Path
from dotenv import load_dotenv
import os



#Create experiment id for mlflow
def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
    else:
      return mlflow.create_experiment(experiment_name)

EXPERIMENT_ID = get_or_create_experiment("Agentic-HPT-Testing") 


# Automatically find and load the .env file
env_path = Path(__file__).resolve().parents[2]/".env"
load_dotenv(dotenv_path=env_path)

# Initialize LLM
API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=API_KEY)


# Define the JSON output parser
json_parser = JsonOutputParser(pydantic_object=ActionOutput)



def initialize_params(state: Parameters) -> TuningState:
    params = {
        "num_layers": state.get("num_layers", 2),
        "learning_rate": state.get("learning_rate", 5e-5),
        "hidden_dim": state.get("hidden_dim", 24),
        "dropout": state.get("dropout", 0.5)
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



def make_objective(fixed_params: Dict, param_to_tune: str):
    def objective(trial):
        # Copy to isolate changes between workers
        params = fixed_params.copy()
        if param_to_tune == "num_layers":
            params["num_layers"] = trial.suggest_int("num_layers", 2, 6)
        elif param_to_tune == "learning_rate":
            params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-2, log=True)
        elif param_to_tune == "hidden_dim":
            params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [24, 48, 96, 144, 192, 256])
        elif param_to_tune == "dropout":
            params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.05, 0.15, 0.25, 0.35, 0.5])
        else:
            raise ValueError(f"Unknown hyperparameter {param_to_tune}")

        # Performs training on model for fixed # of epochs
        # Evaluates results and returns accuracy
        acc = train_and_eval(params)

        #mlflow logging
        with mlflow.start_run(experiment_id=EXPERIMENT_ID):
            mlflow.log_param(param_to_tune, params[param_to_tune])
            mlflow.log_metrics({"accuracy": acc})
            mlflow.set_tag("tuned_param", param_to_tune)
            # # Optionally log all fixed params for traceability
            # for k, v in fixed_params.items():
            #     mlflow.log_param(f"fixed_{k}", v)
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
        "best_score": study.best_value
    }



def coordinator(state: TuningState):
    required_workers = {"num_layers", "learning_rate", "hidden_dim", "dropout"}
    # Type enforcement needed because mlflow's flaw of only returning strings for values
    param_casts = {
        "num_layers": int,
        "hidden_dim": int,
        "learning_rate": float,
        "dropout": float
    }

    workers_done = state["workers_done"]

    # Wait until all workers finish
    if not required_workers.issubset(set(workers_done)):
        # Still waiting for workers
        return Command(update={}, goto="wait")

    iteration = state["iteration"] + 1

    
    runs = mlflow.search_runs(output_format="pandas", experiment_ids=[EXPERIMENT_ID], max_results=20)
    
    # Group by tuned_param and find the best run in each group
    idx = runs.groupby("tags.tuned_param")["metrics.accuracy"].idxmax()
    best_per_param = runs.loc[idx]

    # Find which parameter's best run had the highest accuracy
    best_overall = best_per_param.loc[best_per_param["metrics.accuracy"].idxmax()]
    best_param = best_overall["tags.tuned_param"]

    # Extract the best value for the parameter (params.<param_name>)
    best_val_raw = best_overall[f"params.{best_param}"]
    best_val = param_casts[best_param](best_val_raw)  # type cast
    best_score = best_overall["metrics.accuracy"]


    # Prepare prompt for LLM
    template = """
    You are an autonomous ML tuning coordinator.
    Your job is to decide whether to continue running tuning iterations or finalize the model.

    Current iteration: {iteration}
    Best validation accuracy so far: {best_score:.4f}

    Rules:
    - If accuracy is very high (e.g., greater than 0.95), you must finalize immediately.
    - The number of iterations does not matter if accuracy is already high.
    - Otherwise, continue tuning.

    Respond only in JSON format:
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["iteration", "best_score"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    # Format the prompt
    formatted_prompt = prompt.format(iteration=iteration, best_score=best_score)
     # Call the LLM
    response = llm.invoke(formatted_prompt)
    # Parse the JSON output directly
    decision_data = json_parser.invoke(response)
    decision = decision_data.get("action", "continue")

    
    # Build the new global params & score
    new_params = {**state["params"], best_param: best_val}
    new_state = {
        "status": "finalize" if decision == "finalize" else "tuning",
        "params": new_params,
        "score": best_score,
        "workers_done": [],
        "iteration": iteration
    }

    if decision == "finalize":
        return Command(update=new_state, goto="finalize")
    else:
        # Continue tuning: send to all workers again
        return Command(update=new_state,goto="start_workers")



# Do one last model run
def finalize(state: TuningState):
    params = state["params"]
    final_acc, model = train_and_eval(params, return_model=True)
    print(f"Final model accuracy: {final_acc}")
     
    # Save the model and config
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": params
    }

    save_path = Path(__file__).resolve().parents[2]/"ml"/"final_model.pth"
    torch.save(checkpoint, save_path)
    print("✅ Saved model + config to final_model.pth")

    return {}



def wait(state: TuningState):
    return Command(goto="coordinator")