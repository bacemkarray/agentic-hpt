# Agentic Hyperparameter Tuning

An autonomous hyperparameter tuning framework orchestrated via LangGraph. This system uses parallel agents to optimize different hyperparameters of a neural network, with an LLM-powered coordinator that determines when tuning should stop.

---

## Overview

This project demonstrates a modular and extensible architecture for agent-driven hyperparameter tuning. It leverages:

- **LangGraph** for node orchestration
- **Optuna** for optimization
- **MLflow** for tracking
- **PyTorch** for model training
- **OpenAI API** for agentic decision-making

---

## Architecture

### Parallel Agent Nodes
Each worker specializes in tuning a specific hyperparameter:
- `num_layers`
- `learning_rate`
- `hidden_dim`
- `dropout`

Workers operate in parallel, logging results to MLflow.

### Coordinator Agent
Once all workers finish, a central LLM-based agent:
1. Analyzes logged runs from MLflow
2. Selects the best result
3. Decides to continue tuning or finalize training

This decision is guided by a prompt-enforced policy, where if accuracy exceeds a certain threshold it will finalize the parameters. Otherwise, it loops back to the start_workers node and continues hyperparameter tuning.

---

## Graph Structure

<img width="808" height="511" alt="image" src="https://github.com/user-attachments/assets/2c1fa26c-91aa-4565-99ef-502c1b829942" />

---

## Getting Started

### Prerequisites
- Python 3.11+ recommended
- A valid `OPENAI_API_KEY`
- (Optional) A valid `LANGSMITH_API_KEY` if you want to track runs on LangSmith

### Installation
```bash
git clone https://github.com/<your-org>/agentic-hpt.git
cd agentic-hpt
pip install -r requirements.txt
```

### Usage

You can invoke the graph in two ways:

#### ✅ Run it as a Python script
This directly launches the full tuning loop from the terminal:

```bash
python agent/graph.py
```

Use it with LangGraph Studio (Optional)
If you want to visualize and interact with the graph via LangGraph Studio:

```bash
langgaph dev
```

---


## What's Next
