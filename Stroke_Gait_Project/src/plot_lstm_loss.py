from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results"

train_loss = np.load(
    RESULT_DIR / "lstm_train_loss.npy"
)

val_loss = np.load(
    RESULT_DIR / "lstm_val_loss.npy"
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
    "LSTM Training and Validation Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = (
    RESULT_DIR
    / "lstm_loss_curve.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved to:")
print(save_path)

plt.show()