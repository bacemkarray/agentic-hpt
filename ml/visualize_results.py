import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# import classes
from ml.mlp_core import TunableMLP 

# 1. Load checkpoint (weights + config)
load_path = Path(__file__).resolve().parent/"final_model.pth"
print(load_path)
ckpt = torch.load(load_path)
cfg = ckpt["config"]

# 2. Re-create the model architecture
model = TunableMLP(
    input_dim=13,
    num_layers=cfg["num_layers"],
    hidden_dim=cfg["hidden_dim"],
    dropout=cfg["dropout"]
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Prepare test data
data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_tensor = torch.from_numpy(X_test).float()

# 4. Run predictions
with torch.no_grad():
    preds = model(X_test_tensor).argmax(dim=1).numpy()

# 5. Plot confusion matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Wine Test Set")
plt.show()
