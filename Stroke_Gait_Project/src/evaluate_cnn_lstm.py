from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import get_dataloaders
from models import CNNLSTM


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn_lstm_best.pt"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:")
print(device)

_, _, test_loader = get_dataloaders(
    batch_size=32
)

model = CNNLSTM().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)

        outputs = model(X)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted = torch.argmax(
            probabilities,
            dim=1
        )

        y_true.extend(
            y.numpy()
        )

        y_pred.extend(
            predicted.cpu().numpy()
        )

        y_prob.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_prob
)

print("\nCNN-LSTM TEST RESULTS")
print("--------------------------------")

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1-score:  {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:   {auc:.4f}"
)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Stroke"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "CNN-LSTM Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "cnn_lstm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()