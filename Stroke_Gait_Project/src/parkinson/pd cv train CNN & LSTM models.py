
from pathlib import Path
import argparse
import random
import re

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from pd_models import PDCNN1D, PDLSTM



ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "parkinson"
PROCESSED = DATA_DIR / "processed" / "multitask"
SPLITS = DATA_DIR / "splits" / "cv_5fold"

X_FILE = PROCESSED / "pd_multitask_X_raw.npy"
META_FILE = PROCESSED / "pd_multitask_metadata.csv"
FOLD_FILE = SPLITS / "pd_5fold_participants.csv"

MODEL_DIR = ROOT / "models" / "parkinson" / "cv_compare_89"
RESULT_DIR = ROOT / "results" / "parkinson" / "cv_compare_89"

EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_column(df, possible_names):
    lookup = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    for name in possible_names:
        if name in lookup:
            return lookup[name]

    raise ValueError(
        f"Required column not found.\n"
        f"Expected one of: {possible_names}\n"
        f"Available: {list(df.columns)}"
    )


def encode_label(value):
    text = str(value).strip().lower()

    if text in ["0", "0.0", "healthy", "control", "hc", "normal"]:
        return 0

    if text in [
        "1", "1.0", "pd", "parkinson",
        "parkinsons", "parkinson's", "patient"
    ]:
        return 1

    raise ValueError(f"Unknown class label: {value}")


def make_model(model_name):
    if model_name == "cnn":
        return PDCNN1D().to(DEVICE)

    if model_name == "lstm":
        return PDLSTM().to(DEVICE)

    raise ValueError(f"Unknown model: {model_name}")


def make_loader(x, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64))
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(yb)
        total_correct += (
            logits.argmax(dim=1) == yb
        ).sum().item()
        total_count += len(yb)

    return (
        total_loss / total_count,
        total_correct / total_count
    )


@torch.no_grad()
def predict_probabilities(model, x):
    model.eval()

    loader = DataLoader(
        torch.from_numpy(x.astype(np.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probabilities = []

    for xb in loader:
        logits = model(xb.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(probs.cpu().numpy())

    return np.asarray(probabilities)


def calculate_metrics(df):
    y_true = df["true_label"].to_numpy()
    y_pred = df["prediction"].to_numpy()
    scores = df["pd_probability"].to_numpy()

    result = {
        "participants": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "pd_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, scores)
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    }

    return result


def load_data():
    for file in [X_FILE, META_FILE, FOLD_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    X = np.load(X_FILE, mmap_mode="r")
    meta = pd.read_csv(META_FILE, dtype=str)
    folds = pd.read_csv(FOLD_FILE, dtype=str)

    if X.ndim != 3 or X.shape[1:] != (180, 12):
        raise ValueError(
            f"Expected [windows, 180, 12], got {X.shape}"
        )

    if len(X) != len(meta):
        raise ValueError(
            "Metadata rows do not match X windows."
        )

    meta_id = find_column(
        meta,
        ["participant_id", "participant", "subject_id",
         "subject", "patient_id", "pid"]
    )

    meta_label = find_column(
        meta,
        ["label", "class", "class_label",
         "target", "diagnosis", "group"]
    )

    fold_id = find_column(
        folds,
        ["participant_id", "participant", "subject_id",
         "subject", "patient_id", "pid"]
    )

    fold_number = find_column(
        folds,
        ["fold", "test_fold", "fold_id", "cv_fold"]
    )

    meta["participant_key"] = (
        meta[meta_id].astype(str).str.strip()
    )
    meta["target_label"] = (
        meta[meta_label].map(encode_label)
    )

    folds["participant_key"] = (
        folds[fold_id].astype(str).str.strip()
    )

    folds["test_fold_number"] = (
        folds[fold_number].astype(str)
        .str.extract(r"(\d+)")[0]
    )

    if folds["test_fold_number"].isna().any():
        raise ValueError("Invalid fold numbers.")

    folds["test_fold_number"] = (
        folds["test_fold_number"].astype(int)
    )

    fold_lookup = folds[
        ["participant_key", "test_fold_number"]
    ].drop_duplicates()

    if fold_lookup["participant_key"].duplicated().any():
        raise ValueError(
            "A participant belongs to multiple test folds."
        )

    meta = meta.merge(
        fold_lookup,
        on="participant_key",
        how="left",
        validate="many_to_one",
        sort=False
    )

    if meta["test_fold_number"].isna().any():
        raise ValueError(
            "Some participants are missing from fold CSV."
        )

    if meta.groupby("participant_key")[
        "target_label"
    ].nunique().max() != 1:
        raise ValueError(
            "Conflicting labels for a participant."
        )

    print("Dataset loaded:", X.shape)
    print(
        "Participants:",
        meta["participant_key"].nunique()
    )

    print(
        "Classes:",
        meta.drop_duplicates("participant_key")[
            "target_label"
        ].value_counts().to_dict()
    )

    return X, meta


def train_one_fold(model_name, fold, X, meta):
    print(f"\n{'=' * 55}")
    print(f"MODEL: {model_name.upper()} | FOLD: {fold}")
    print(f"{'=' * 55}")

    set_seed(SEED + fold)

    output_models = MODEL_DIR / model_name
    output_results = RESULT_DIR / model_name

    output_models.mkdir(parents=True, exist_ok=True)
    output_results.mkdir(parents=True, exist_ok=True)

    checkpoint = output_models / f"fold_{fold}_best.pt"
    predictions_file = (
        output_results / f"fold_{fold}_participants.csv"
    )

    if checkpoint.exists() or predictions_file.exists():
        if checkpoint.exists() and predictions_file.exists():
            print("Already completed. Skipping fold.")
            return pd.read_csv(predictions_file)

        raise FileExistsError(
            f"Partial previous output exists for fold {fold}.\n"
            "Check the files before restarting."
        )

    test_mask = (
        meta["test_fold_number"].to_numpy() == fold
    )

    remaining = meta.loc[~test_mask].copy()
    test_meta = meta.loc[test_mask].copy()

    people = (
        remaining[
            ["participant_key", "target_label"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_people, val_people = train_test_split(
        people["participant_key"].to_numpy(),
        test_size=0.15,
        random_state=SEED + fold,
        stratify=people["target_label"].to_numpy()
    )

    train_mask = (
        meta["participant_key"].isin(train_people).to_numpy()
    )

    val_mask = (
        meta["participant_key"].isin(val_people).to_numpy()
    )

    x_train = np.asarray(X[train_mask], dtype=np.float32)
    x_val = np.asarray(X[val_mask], dtype=np.float32)
    x_test = np.asarray(X[test_mask], dtype=np.float32)

    y_train = meta.loc[
        train_mask, "target_label"
    ].to_numpy(dtype=np.int64)

    y_val = meta.loc[
        val_mask, "target_label"
    ].to_numpy(dtype=np.int64)

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    print(
        f"Windows: train={len(x_train)}, "
        f"val={len(x_val)}, test={len(x_test)}"
    )

    print(
        "Test participants:",
        test_meta["participant_key"].nunique()
    )

    train_loader = make_loader(
        x_train, y_train, shuffle=True
    )
    val_loader = make_loader(
        x_val, y_val, shuffle=False
    )

    model = make_model(model_name)

    counts = np.bincount(
        y_train, minlength=2
    ).astype(np.float32)

    if np.any(counts == 0):
        raise ValueError(
            "Training data has a missing class."
        )

    class_weights = (
        len(y_train) / (2.0 * counts)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    waiting = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
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
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            waiting = 0

            torch.save(
                model.state_dict(),
                checkpoint
            )

            np.save(
                output_models / f"fold_{fold}_mean.npy",
                mean
            )

            np.save(
                output_models / f"fold_{fold}_std.npy",
                std
            )

        else:
            waiting += 1

        if waiting >= PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        output_results / f"fold_{fold}_history.csv",
        index=False
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    probabilities = predict_probabilities(
        model, x_test
    )

    window_results = test_meta[
        ["participant_key", "target_label"]
    ].copy()

    window_results["pd_probability"] = probabilities

    # Average window scores within each participant.
    participant_results = (
        window_results
        .groupby("participant_key", as_index=False)
        .agg(
            true_label=("target_label", "first"),
            pd_probability=("pd_probability", "mean"),
            windows=("pd_probability", "size")
        )
    )

    participant_results["prediction"] = (
        participant_results["pd_probability"] >= 0.5
    ).astype(int)

    participant_results["fold"] = fold
    participant_results["model"] = model_name

    participant_results.to_csv(
        predictions_file,
        index=False
    )

    metrics = calculate_metrics(participant_results)

    print(
        f"Fold {fold} Participant Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    return participant_results

def evaluate_model(model_name, X, meta):
    all_results = []

    available_folds = sorted(
        meta["test_fold_number"].unique()
    )

    if len(available_folds) != 5:
        raise ValueError(
            f"Expected 5 folds, found {available_folds}"
        )

    for fold in available_folds:
        result = train_one_fold(
            model_name, int(fold), X, meta
        )
        all_results.append(result)

    combined = pd.concat(
        all_results, ignore_index=True
    )

    if combined["participant_key"].duplicated().any():
        raise ValueError(
            "Duplicate test participant in predictions."
        )

    expected = meta["participant_key"].nunique()

    if len(combined) != expected:
        raise ValueError(
            f"Expected {expected} participants, "
            f"got {len(combined)}"
        )

    out = RESULT_DIR / model_name

    combined.to_csv(
        out / "all_participant_predictions.csv",
        index=False
    )

    metrics = calculate_metrics(combined)

    matrix = confusion_matrix(
        combined["true_label"],
        combined["prediction"],
        labels=[0, 1]
    )

    print(f"\nFINAL {model_name.upper()} RESULTS")
    print("-" * 45)

    for key, value in metrics.items():
        if key == "participants":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    pd.DataFrame([{
        "model": model_name,
        **metrics
    }]).to_csv(
        out / "final_metrics.csv",
        index=False
    )

    return {
        "model": model_name,
        **metrics
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["cnn", "lstm"],
        default=["cnn", "lstm"]
    )

    args = parser.parse_args()

    set_seed(SEED)
    torch.set_num_threads(4)

    X, meta = load_data()

    if meta["participant_key"].nunique() != 89:
        raise ValueError(
            "Expected 89 participants. "
            "Check the participant IDs and data."
        )

    summary = []

    for model_name in args.models:
        result = evaluate_model(
            model_name, X, meta
        )
        summary.append(result)

    RESULT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    pd.DataFrame(summary).to_csv(
        RESULT_DIR / "cnn_lstm_new_model_summary.csv",
        index=False
    )

    print("\nTraining and evaluation completed.")
    print("Results saved to:", RESULT_DIR)
    print("Existing CNN-LSTM models were not modified.")


if __name__ == "__main__":
    main()
de32