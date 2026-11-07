from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from models import CNN1D



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE")
print("------------------------------")
print(device)


# -------------------------------------------------
# DATA
# -------------------------------------------------

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)


y_train = np.load(
    DATA_DIR / "train_y.npy"
)

healthy_count = np.sum(
    y_train == 0
)

stroke_count = np.sum(
    y_train == 1
)

total_count = len(y_train)

healthy_weight = (
    total_count
    /
    (2 * healthy_count)
)

stroke_weight = (
    total_count
    /
    (2 * stroke_count)
)

class_weights = torch.tensor(
    [
        healthy_weight,
        stroke_weight
    ],
    dtype=torch.float32
).to(device)


print("\nCLASS COUNTS")
print("------------------------------")

print(
    "Healthy windows:",
    healthy_count
)

print(
    "Stroke windows:",
    stroke_count
)

print("\nClass weights:")

print(
    "Healthy:",
    healthy_weight
)

print(
    "Stroke:",
    stroke_weight
)


model = CNN1D().to(
    device
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_loss = float("inf")

patience = 5
patience_counter = 0

best_model_path = (
    MODEL_DIR
    / "cnn_best.pt"
)


train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


for epoch in range(EPOCHS):


    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += y.size(0)

        correct += (
            predicted == y
        ).sum().item()


    train_loss = (
        running_loss
        / total
    )

    train_accuracy = (
        correct
        / total
    )


    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            val_running_loss += (
                loss.item()
                * X.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += y.size(0)

            val_correct += (
                predicted == y
            ).sum().item()


    val_loss = (
        val_running_loss
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    )


    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        patience_counter = 0

        saved_text = " <-- BEST SAVED"

    else:

        patience_counter += 1

        saved_text = ""


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
        f"{saved_text}"
    )


    if patience_counter >= patience:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )

        break


np.save(
    RESULT_DIR
    / "cnn_train_loss.npy",
    np.array(train_losses)
)

np.save(
    RESULT_DIR
    / "cnn_val_loss.npy",
    np.array(val_losses)
)

np.save(
    RESULT_DIR
    / "cnn_train_accuracy.npy",
    np.array(train_accuracies)
)

np.save(
    RESULT_DIR
    / "cnn_val_accuracy.npy",
    np.array(val_accuracies)
)


print("\n================================")
print("CNN TRAINING COMPLETE")
print("================================")

print(
    "Best model saved to:"
)

print(
    best_model_path
)

print(
    "\nBest validation loss:"
)

print(
    best_val_loss
)