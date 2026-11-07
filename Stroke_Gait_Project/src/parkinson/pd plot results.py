
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "parkinson"
RESULTS.mkdir(parents=True, exist_ok=True)

models = {
    "1D CNN": "pd_cnn",
    "LSTM": "pd_lstm",
    "CNN-LSTM": "pd_cnn_lstm"
}

for model_name, prefix in models.items():

    train_loss = np.load(RESULTS / f"{prefix}_train_loss.npy")
    val_loss = np.load(RESULTS / f"{prefix}_val_loss.npy")
    train_acc = np.load(RESULTS / f"{prefix}_train_accuracy.npy")
    val_acc = np.load(RESULTS / f"{prefix}_val_accuracy.npy")

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Training")
    plt.plot(epochs, val_loss, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{prefix}_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc * 100, marker="o", label="Training")
    plt.plot(epochs, val_acc * 100, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{prefix}_accuracy_curve.png", dpi=300)
    plt.close()

    print(f"Saved loss and accuracy curves: {model_name}")


comparison = pd.read_csv(RESULTS / "pd_model_comparison.csv")

participant = comparison[
    comparison["Level"] == "Participant"
].set_index("Model")

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, model_name in enumerate(models):
    values = participant.loc[model_name, metrics].to_numpy(dtype=float)
    ax.bar(
        x + (i - 1) * width,
        values * 100,
        width,
        label=model_name
    )

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "PD Recall", "F1", "ROC-AUC"])
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_title("Parkinson's Classification - Participant-Level Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    RESULTS / "pd_participant_model_comparison.png",
    dpi=300
)
plt.close()

print("Location:", RESULTS)
