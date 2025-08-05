# Agentic Hyperparameter Tuning

An autonomous hyperparameter tuning framework orchestrated via LangGraph. This system uses nodes working in parallel to optimize different hyperparameters of a neural network, with an LLM-powered coordinator that determines when tuning should stop.

---

## Overview

This project demonstrates a modular and extensible architecture for agent-driven hyperparameter tuning. It leverages:

- **LangGraph** for node orchestration and communication
- **PyTorch** for model training and eval
- **Optuna** for optimization
- **MLflow** for tracking

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
git clone https://github.com/bacemkarray/agentic-hpt.git
cd agentic-hpt
pip install -r requirements.txt
```

### Usage

You can invoke the graph in two ways:

#### Run it as a Python script
This directly launches the full tuning loop from the terminal:

```bash
python agent/graph.py
```

Use it with LangGraph Studio (Optional)
If you want to visualize and interact with the graph via LangGraph Studio:

```bash
langgraph dev
```

---

## What's Next

While this project already demonstrates a modular, agentic tuning pipeline, there are a few extensions that could evolve it further:

### Distributed Multi-Objective Optimization
The current system uses per-parameter parallelism, which is modular but optimizes each hyperparameter in isolation. This can miss interdependencies. For example, the ideal `learning_rate` may depend on the selected `num_layers`. A future version could use Optuna’s joint optimization capabilities (e.g. via `Optuna's RDB backend`) to tune parameters in combination, capturing cross-parameter effects and better exploring the joint search space. This would also enable training deeper models or working with more complex datasets, where tuning sensitivity and parameter interactions becomes more critical.

### Dynamic Parameter Choices
The current graph statically defines one node per parameter, which is straightforward but inflexible. A more dynamic architecture can utilize LangGraph’s Send API to spawn multiple worker nodes at runtime, each processing different parameters in parallel. This enables an upstream node to programmatically determine which parameters to tune based on factors such as dataset complexity or model behavior, and then dispatch corresponding tasks dynamically, supporting scalable and adaptive workflows through LangGraph’s map-reduce style execution.

### Smarter Coordinator Agent
The coordinator currently makes stop decisions based on current best accuracy. Future prompts could incorporate trend analysis, time-based constraints, or even allow the LLM to recommend which parameters to freeze or prioritize next - expanding agentic behavior beyond binary decisions.

---

These highlight just how much this project could evolve - but for now, this system stands on its own as a complete and meaningful demonstration of orchestrated tuning, intelligent stopping, and agentic control in ML pipelines. 

I really enjoyed building this project. It taught me the importance of proper scoping. Without clear boundaries, it's easy to fall into the trap of endlessly chasing a “perfect version” that keeps evolving in your head, raising the bar higher every time you get close to finishing. It's a lesson said by most but one I had to learn through experience: shipping SOMETHING is better than keeping things under development.



