from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pd_models import PDCNNLSTM

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "parkinson" / "processed" / "multitask"
CV_DIR = ROOT / "data" / "parkinson" / "splits" / "cv_5fold"

MODEL_DIR = ROOT / "models" / "parkinson" / "cv_multitask"
RESULT_DIR = ROOT / "results" / "parkinson" / "cv_multitask"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X = np.load(DATA_DIR / "pd_multitask_X_raw.npy")

meta = pd.read_csv(
    CV_DIR / "pd_5fold_window_metadata.csv",
    dtype={"participant_id": str}
)

assert X.shape[1:] == (180, 12)
assert len(X) == len(meta)
assert meta["fold"].between(1, 5).all()

participants = (
    meta[["participant_id", "label", "fold"]]
    .drop_duplicates()
)

assert len(participants) == 89
assert participants["participant_id"].is_unique

print("PARKINSON 5-FOLD CNN-LSTM TRAINING")
print("=" * 60)
print("Participants:", len(participants))
print("Windows:", len(X))
print("Device:", DEVICE)


def make_loader(indices, mean, std, shuffle):

    data = (X[indices] - mean) / std

    inputs = torch.tensor(
        data, dtype=torch.float32
    )

    labels = torch.tensor(
        meta.iloc[indices]["label"].to_numpy(),
        dtype=torch.long
    )

    dataset = TensorDataset(inputs, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


def run_epoch(model, loader, criterion, optimiser=None):

    training = optimiser is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    count = 0

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimiser.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimiser.step()

        batch_size = len(labels)

        total_loss += loss.item() * batch_size
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        count += batch_size

    return total_loss / count, correct / count


for fold in range(1, 6):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}/5")
    print("=" * 60)

    # Participant-level held-out test fold
    test_people = set(
        participants.loc[
            participants["fold"] == fold,
            "participant_id"
        ]
    )

    remaining = participants[
        participants["fold"] != fold
    ].copy()

    # Inner validation: participant-level split
    train_people_df, val_people_df = train_test_split(
        remaining,
        test_size=0.15,
        stratify=remaining["label"],
        random_state=SEED + fold
    )

    train_people = set(train_people_df["participant_id"])
    val_people = set(val_people_df["participant_id"])

    assert train_people.isdisjoint(val_people)
    assert train_people.isdisjoint(test_people)
    assert val_people.isdisjoint(test_people)

    ids = meta["participant_id"]

    train_idx = np.flatnonzero(ids.isin(train_people))
    val_idx = np.flatnonzero(ids.isin(val_people))
    test_idx = np.flatnonzero(ids.isin(test_people))

    print("Train participants:", len(train_people))
    print("Val participants:", len(val_people))
    print("Test participants:", len(test_people))

    print("Train windows:", len(train_idx))
    print("Val windows:", len(val_idx))
    print("Test windows:", len(test_idx))

    # Normalisation calculated from training only
    mean = X[train_idx].mean(
        axis=(0, 1), keepdims=True
    )

    std = X[train_idx].std(
        axis=(0, 1), keepdims=True
    )

    std = np.where(std < 1e-6, 1.0, std)

    np.save(
        MODEL_DIR / f"fold_{fold}_mean.npy",
        mean
    )

    np.save(
        MODEL_DIR / f"fold_{fold}_std.npy",
        std
    )

    train_loader = make_loader(
        train_idx, mean, std, True
    )

    val_loader = make_loader(
        val_idx, mean, std, False
    )

    test_loader = make_loader(
        test_idx, mean, std, False
    )

    # Class weights from training windows only
    y_train = meta.iloc[train_idx]["label"].to_numpy()

    counts = np.bincount(y_train, minlength=2)

    weights = len(y_train) / (2.0 * counts)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    torch.manual_seed(SEED + fold)

    model = PDCNNLSTM().to(DEVICE)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    checkpoint = MODEL_DIR / f"fold_{fold}_best.pt"

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimiser
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss - 1e-6:

            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                model.state_dict(),
                checkpoint
            )

        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping")
            break

    pd.DataFrame(history).to_csv(
        RESULT_DIR / f"fold_{fold}_history.csv",
        index=False
    )


    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()
    probabilities = []

    with torch.no_grad():

        for inputs, _ in test_loader:

            outputs = model(inputs.to(DEVICE))

            probs = torch.softmax(
                outputs, dim=1
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    assert len(probabilities) == len(test_idx)


    predictions = meta.iloc[test_idx].copy()

    predictions["window_index"] = test_idx
    predictions["pd_probability"] = probabilities
    predictions["prediction"] = (
        predictions["pd_probability"] >= 0.5
    ).astype(int)

    predictions.to_csv(
        RESULT_DIR / f"fold_{fold}_windows.csv",
        index=False
    )

    recording_scores = (
        predictions.groupby(
            ["participant_id", "recording_id", "label"],
            as_index=False
        )["pd_probability"].mean()
    )

    participant_scores = (
        recording_scores.groupby(
            ["participant_id", "label"],
            as_index=False
        )["pd_probability"].mean()
    )

    participant_scores["prediction"] = (
        participant_scores["pd_probability"] >= 0.5
    ).astype(int)

    participant_scores["fold"] = fold

    participant_scores.to_csv(
        RESULT_DIR / f"fold_{fold}_participants.csv",
        index=False
    )

    correct = (
        participant_scores["prediction"]
        == participant_scores["label"]
    ).sum()

    print(
        f"\nFold {fold} finished | "
        f"Best epoch: {best_epoch} | "
        f"Participant accuracy: "
        f"{correct / len(participant_scores) * 100:.2f}%"
    )

print("\n" + "=" * 60)
print("ALL FIVE FOLDS COMPLETED")
print("Saved models:", MODEL_DIR)
print("Saved predictions:", RESULT_DIR)
