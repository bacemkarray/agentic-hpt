import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── 1. Data Prep (Wine) ──────────────────────────────────────────────────────

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = torch.from_numpy(X_train).float()
X_val   = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).long()
y_val   = torch.from_numpy(y_val).long()

def get_loader(X, y, batch_size):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# ─── 2. Dynamic MLP Definition ────────────────────────────────────────────────

class TunableMLP(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))  # 3 classes in Wine
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─── 3. Train/Eval Helper ─────────────────────────────────────────────────────

def train_and_eval(cfg):
    model = TunableMLP(
        input_dim=13,
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"]
    )
    opt = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = get_loader(X_train, y_train, cfg["batch_size"])
    val_loader   = get_loader(X_val,   y_val,   cfg["batch_size"])

    # fixed-epoch training
    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    # final evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / total

# ─── 4. Expanded Search Space ─────────────────────────────────────────────────

search_space = {
    "num_layers":   [1, 2, 3, 4],
    "hidden_dim":   [16, 32, 64, 128],
    "dropout":      [0.0, 0.1, 0.3, 0.5],
    "lr":           [1e-2, 1e-3, 1e-4],
    "batch_size":   [16, 32, 64],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

# ─── 5. Baseline & Tuning Loop ────────────────────────────────────────────────

base_cfg = {
    "num_layers":   1,
    "hidden_dim":   64,
    "dropout":      0.9,
    "lr":           1e-7,
    "batch_size":   32,
    "weight_decay": 1e-2,
    "epochs":       50,
}

baseline_acc = train_and_eval(base_cfg)
print(f"Round 0 — baseline acc: {baseline_acc:.4f}\n")

max_rounds      = 5
trials_per_param = 5

for rnd in range(1, max_rounds + 1):
    print(f"=== Round {rnd} ===")
    best_improve = 0.0
    best_param   = None
    best_val     = None
    best_acc     = baseline_acc

    for param, choices in search_space.items():
        cfg_snapshot = copy.deepcopy(base_cfg)

        def objective(trial):
            val = trial.suggest_categorical(param, choices)
            cfg_snapshot[param] = val
            return train_and_eval(cfg_snapshot)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials_per_param)

        acc   = study.best_value
        val   = study.best_params[param]
        delta = acc - baseline_acc
        print(f" • {param:10s} → best={val:<6} acc={acc:.4f} Δ={delta:+.4f}")

        if delta > best_improve:
            best_improve = delta
            best_param   = param
            best_val     = val
            best_acc     = acc

    if best_improve <= 0:
        print("→ No improvement this round; stopping.\n")
        break

    base_cfg[best_param] = best_val
    baseline_acc         = best_acc
    print(f"→ Committed {best_param} → {best_val}, new baseline acc={baseline_acc:.4f}\n")

# ─── 6. Final Report ─────────────────────────────────────────────────────────

print("=== Final Training ===")
final_acc = train_and_eval(base_cfg)
print(f"Final config: {base_cfg}")
print(f"Final accuracy: {final_acc:.4f}")

print("=== Final Training ===")
# Re-train and capture the model object
final_model = TunableMLP(
    input_dim=13,
    num_layers=base_cfg["num_layers"],
    hidden_dim=base_cfg["hidden_dim"],
    dropout=base_cfg["dropout"]
)
opt = optim.Adam(final_model.parameters(),
                 lr=base_cfg["lr"],
                 weight_decay=base_cfg["weight_decay"])
loss_fn = nn.CrossEntropyLoss()
train_loader = get_loader(X_train, y_train, base_cfg["batch_size"])
for epoch in range(base_cfg["epochs"]):
    final_model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        loss_fn(final_model(xb), yb).backward()
        opt.step()

checkpoint = {
    "model_state_dict": final_model.state_dict(),
    "config": base_cfg
}
torch.save(checkpoint, "final_model.pth")
print("Saved model + config to final_model.pth")