from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results"


train_loss = np.load(
    RESULT_DIR / "cnn_train_loss.npy"
)

val_loss = np.load(
    RESULT_DIR / "cnn_val_loss.npy"
)

train_accuracy = np.load(
    RESULT_DIR / "cnn_train_accuracy.npy"
)

val_accuracy = np.load(
    RESULT_DIR / "cnn_val_accuracy.npy"
)


epochs = np.arange(
    1,
    len(train_loss) + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "1D CNN Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "cnn_loss_curve.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracy * 100,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy * 100,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "1D CNN Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "cnn_accuracy_curve.png",
    dpi=300
)

plt.show()


print("CNN GRAPHS CREATED")
print("------------------------------")

print("Saved:")
print(
    RESULT_DIR / "cnn_loss_curve.png"
)

print(
    RESULT_DIR / "cnn_accuracy_curve.png"
)

