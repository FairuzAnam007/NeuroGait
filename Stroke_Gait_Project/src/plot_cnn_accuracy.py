from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results"

train_accuracy = np.load(
    RESULT_DIR / "cnn_train_accuracy.npy"
)

val_accuracy = np.load(
    RESULT_DIR / "cnn_val_accuracy.npy"
)

epochs = np.arange(
    1,
    len(train_accuracy) + 1
)

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

save_path = (
    RESULT_DIR / "cnn_accuracy_curve.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved to:")
print(save_path)

plt.show()