from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pd_dataset import get_pd_dataloaders
from pd_models import PDLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "parkinson" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "parkinson"
RESULT_DIR = PROJECT_ROOT / "results" / "parkinson"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5

print("DEVICE:", DEVICE)


train_loader, val_loader, _ = get_pd_dataloaders(
    batch_size=BATCH_SIZE
)


train_y = np.load(DATA_DIR / "pd_train_y.npy")

counts = np.bincount(train_y, minlength=2)

if np.any(counts == 0):
    raise ValueError("Both Healthy and Parkinson classes are required")

weights = len(train_y) / (2.0 * counts)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE
)

print("Healthy windows:", counts[0])
print("Parkinson windows:", counts[1])
print("Class weights:", weights)


model = PDLSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

BEST_PATH = MODEL_DIR / "pd_lstm_best.pt"


for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += len(y)

    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            loss = criterion(outputs, y)

            total_loss += loss.item() * len(y)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += len(y)

    val_loss = total_loss / total
    val_acc = correct / total


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)


    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}%"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            BEST_PATH
        )

        patience_counter = 0

        print("Best LSTM model saved")

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break

np.save(
    RESULT_DIR / "pd_lstm_train_loss.npy",
    np.array(train_losses)
)

np.save(
    RESULT_DIR / "pd_lstm_val_loss.npy",
    np.array(val_losses)
)

np.save(
    RESULT_DIR / "pd_lstm_train_accuracy.npy",
    np.array(train_accuracies)
)

np.save(
    RESULT_DIR / "pd_lstm_val_accuracy.npy",
    np.array(val_accuracies)
)


print("\n" + "=" * 50)
print("PD LSTM TRAINING COMPLETED")
print("=" * 50)

print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

print("Best model saved:")
print(BEST_PATH)

print("\nTraining history saved in:")
print(RESULT_DIR)
