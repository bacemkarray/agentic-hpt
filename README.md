# Agentic Hyperparameter Tuning (LangGraph + PyTorch + Optuna)

An autonomous hyperparameter tuning framework orchestrated via LangGraph. This system uses parallel agents to optimize different hyperparameters of a neural network, with an LLM-powered coordinator that determines when tuning should stop.

---

## 🚀 Overview

This project demonstrates a modular and extensible architecture for agent-driven hyperparameter tuning. It leverages:

- **LangGraph** for node orchestration
- **Optuna** for optimization
- **MLflow** for tracking
- **PyTorch** for model training
- **OpenAI API** for agentic decision-making

---

## 🧠 Architecture

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

This decision is guided by a prompt-enforced policy:
- If accuracy > 0.95 → finalize
- Otherwise → continue

---

## 🧩 Graph Structure

<img width="815" height="497" alt="image" src="https://github.com/user-attachments/assets/d8f421ab-1e90-4f1a-adfb-21c6b9939341" />

<img width="808" height="511" alt="image" src="https://github.com/user-attachments/assets/2c1fa26c-91aa-4565-99ef-502c1b829942" />
