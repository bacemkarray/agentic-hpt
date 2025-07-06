import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── 1. Data Prep ─────────────────────────────────────────────────────────────

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = torch.from_numpy(X_train).float()
X_val   = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).long()
y_val   = torch.from_numpy(y_val).long()

# Use DataLoader for mini-batches
def get_loader(X, y, batch_size):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# ─── 2. Model Definition ──────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, x):
        return self.net(x)

# ─── 3. Train/Eval with Early-Stopping & Scheduler ────────────────────────────

def train_and_eval(cfg):
    # Build model & optimizer
    model = SimpleNet(4, cfg["hidden_dim"], cfg["dropout"])
    opt   = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    # LR scheduler
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=cfg["step_size"], gamma=cfg["gamma"])
    loss_fn = nn.CrossEntropyLoss()

    # DataLoaders
    train_loader = get_loader(X_train, y_train, cfg["batch_size"])
    val_loader   = get_loader(X_val, y_val, cfg["batch_size"])

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss_fn(out, yb).backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total

        # Early-stopping
        if val_acc > best_val_acc + cfg["es_min_delta"]:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["es_patience"]:
                break

    return best_val_acc

# ─── 4. Expanded Search Space ─────────────────────────────────────────────────

search_space = {
    "lr":           [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "dropout":      [0.0, 0.1, 0.3, 0.5],
    "hidden_dim":   [16, 32, 64, 128],
    "batch_size":   [8, 16, 32, 64],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "step_size":    [5, 10, 20],
    "gamma":        [0.1, 0.5, 0.9],
    # early-stop params are fixed, but you could tune these too
}

# ─── 5. Baseline Config & Eval ─────────────────────────────────────────────────

base_cfg = {
    "lr":           1e-4,      # too low, slow/no learning
    "dropout":      0.5,       # too high, lots of info lost
    "hidden_dim":   8,         # small capacity
    "batch_size":   64,        # might underfit with few updates
    "weight_decay": 1e-2,      # too much regularization
    "step_size":    2,         # over-aggressive LR drops
    "gamma":        0.1,       # sharp decay
    "epochs":       50,        # shorter window
    "es_patience":  3,         # quick to quit
    "es_min_delta": 0.01       # requires large improvement
}

baseline_acc = train_and_eval(base_cfg)
print(f"Round 0 — baseline acc: {baseline_acc:.4f}\n")

# ─── 6. Agentic Tuning Loop w/ Pruning ────────────────────────────────────────

max_rounds      = 2
trials_per_param = 10

for rnd in range(1, max_rounds + 1):
    print(f"=== Round {rnd} ===")
    best_improve = 0.0
    best_param   = None
    best_val     = None
    best_acc     = baseline_acc

    for param, choices in search_space.items():
        # snapshot config for this param
        cfg_snapshot = copy.deepcopy(base_cfg)

        def objective(trial, cfg=cfg_snapshot, p=param):
            # Suggest hyperparam value
            val = trial.suggest_categorical(p, choices)
            cfg[p] = val

            # Use a median pruner to cut bad trials early
            trial.set_user_attr("cfg", cfg)
            acc = train_and_eval(cfg)
            trial.report(acc, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return acc

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=trials_per_param, catch=(optuna.exceptions.TrialPruned,))

        acc = study.best_value
        val = study.best_params[param]
        delta = acc - baseline_acc

        print(f" • {param:12s} → best={val:<6} acc={acc:.4f} Δ={delta:+.4f}")

        if delta > best_improve:
            best_improve = delta
            best_param   = param
            best_val     = val
            best_acc     = acc

    if best_improve <= 0:
        print("→ No further improvement; halting.\n")
        break

    # commit the winning change
    base_cfg[best_param] = best_val
    baseline_acc = best_acc
    print(f"→ Tuning {best_param} → {best_val}, new baseline acc={baseline_acc:.4f}\n")

# ─── 7. Final Report ───────────────────────────────────────────────────────────

print("=== Final Training ===")
final_acc = train_and_eval(base_cfg)
print(f"Final config: {base_cfg}")
print(f"Final accuracy: {final_acc:.4f}")