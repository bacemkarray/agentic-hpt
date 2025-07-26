import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dataprep
data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).long()
y_val = torch.from_numpy(y_val).long()


# MLP Definition
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
        layers.append(nn.Linear(in_dim, 3))  # output layer for 3 wine classes
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Train and eval helper
def train_and_eval(cfg, return_model=False):
    model = TunableMLP(
        input_dim=13,
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"]
    )
    opt = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"]
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size=32,
                              shuffle=True)
    
    val_loader = DataLoader(TensorDataset(X_val, y_val), 
                              batch_size=32,
                              shuffle=True)

    EPOCHS = 50
    # fixed-epoch training
    for epoch in range(EPOCHS):
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
    
    if return_model:
        return correct/total, model
    return correct / total