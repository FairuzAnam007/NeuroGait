from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pd_models import PDCNNLSTM

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "parkinson"
    / "processed" / "multitask"
)

MODEL_DIR = (
    ROOT / "models" / "parkinson"
    / "final_multitask"
)

RESULT_DIR = (
    ROOT / "results" / "parkinson"
    / "final_multitask"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

X = np.load(
    DATA_DIR / "pd_multitask_X_raw.npy"
).astype(np.float32)

meta = pd.read_csv(
    DATA_DIR / "pd_multitask_metadata.csv",
    dtype={"participant_id": str}
)

assert X.shape == (5072, 180, 12)
assert len(meta) == len(X)
assert meta["participant_id"].nunique() == 89
assert np.isfinite(X).all()

y = meta["label"].to_numpy(dtype=np.int64, copy=True)

print("FINAL PARKINSON CNN-LSTM TRAINING")
print("=" * 55)
print("Participants:", meta["participant_id"].nunique())
print("Windows:", len(X))
print("Device:", device)

mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True)

std = np.where(std < 1e-6, 1.0, std)

X_normalised = (X - mean) / std

np.save(
    MODEL_DIR / "pd_final_mean.npy",
    mean
)

np.save(
    MODEL_DIR / "pd_final_std.npy",
    std
)

dataset = TensorDataset(
    torch.from_numpy(X_normalised).float(),
    torch.from_numpy(y).long()
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

counts = np.bincount(y, minlength=2)

weights = len(y) / (2.0 * counts)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )
)

model = PDCNNLSTM().to(device)

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total += len(labels)

    train_loss = total_loss / total
    train_accuracy = correct / total

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}%",
        flush=True
    )

checkpoint = (
    MODEL_DIR / "pd_final_cnn_lstm.pt"
)

torch.save(model.state_dict(), checkpoint)

pd.DataFrame(history).to_csv(
    RESULT_DIR / "pd_final_training_history.csv",
    index=False
)

print("\nFINAL PD MODEL SAVED")
print("Checkpoint:", checkpoint)
print("Mean:", MODEL_DIR / "pd_final_mean.npy")
print("Std:", MODEL_DIR / "pd_final_std.npy")

