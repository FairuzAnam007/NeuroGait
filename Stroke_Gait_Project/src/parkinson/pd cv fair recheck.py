from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

DATA = ROOT / "data" / "parkinson"
RESULTS = ROOT / "results" / "parkinson"

MODEL_DIR = (
    ROOT / "models" / "parkinson" / "cv_compare_89"
)

OLD_RESULTS = RESULTS / "cv_multitask"

OUTPUT = RESULTS / "cv_compare_89" / "fair_recheck"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

MODEL_CLASSES = {
    "1D CNN": ("cnn", PDCNN1D),
    "LSTM": ("lstm", PDLSTM),
}

X = np.load(
    DATA / "processed" / "multitask"
    / "pd_multitask_X_raw.npy",
    mmap_mode="r"
)

meta = pd.read_csv(
    DATA / "splits" / "cv_5fold"
    / "pd_5fold_window_metadata.csv",
    dtype={"participant_id": str}
)

required = [
    "participant_id", "recording_id",
    "label", "fold"
]

for column in required:
    if column not in meta.columns:
        raise ValueError(
            f"Missing metadata column: {column}"
        )

assert X.shape == (5072, 180, 12)
assert len(meta) == len(X)
assert meta["participant_id"].nunique() == 89

@torch.no_grad()
def get_probabilities(model, data):

    model.eval()
    output = []

    for start in range(0, len(data), BATCH_SIZE):

        batch = torch.tensor(
            data[start:start + BATCH_SIZE],
            dtype=torch.float32,
            device=DEVICE
        )

        logits = model(batch)

        probabilities = torch.softmax(
            logits, dim=1
        )[:, 1]

        output.extend(
            probabilities.cpu().numpy().tolist()
        )

    return np.asarray(output)


def evaluate_saved_model(model_name, folder, model_class):

    all_participants = []

    for fold in range(1, 6):

        print(f"{model_name}: Fold {fold}")

        fold_model_dir = MODEL_DIR / folder

        checkpoint = (
            fold_model_dir / f"fold_{fold}_best.pt"
        )

        mean_file = (
            fold_model_dir / f"fold_{fold}_mean.npy"
        )

        std_file = (
            fold_model_dir / f"fold_{fold}_std.npy"
        )

        for file in [checkpoint, mean_file, std_file]:
            if not file.exists():
                raise FileNotFoundError(file)

        test_indices = np.flatnonzero(
            meta["fold"].to_numpy() == fold
        )

        # Check test participants match the
        # original saved CNN/LSTM evaluation.
        saved_file = (
            RESULTS / "cv_compare_89" / folder
            / f"fold_{fold}_participants.csv"
        )

        if not saved_file.exists():
            raise FileNotFoundError(saved_file)

        saved = pd.read_csv(saved_file, dtype=str)

        expected_ids = set(
            meta.iloc[test_indices][
                "participant_id"
            ].astype(str)
        )

        saved_ids = set(
            saved["participant_key"].astype(str)
        )

        if expected_ids != saved_ids:
            raise ValueError(
                f"{model_name} fold {fold}: "
                "Test participant mismatch."
            )

        mean = np.load(mean_file)
        std = np.load(std_file)

        test_data = (
            np.asarray(X[test_indices], dtype=np.float32)
            - mean
        ) / std

        model = model_class().to(DEVICE)

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=DEVICE,
                weights_only=True
            )
        )

        probabilities = get_probabilities(
            model, test_data
        )

        window_predictions = (
            meta.iloc[test_indices].copy()
        )

        window_predictions["pd_probability"] = (
            probabilities
        )

        # SAME aggregation as original CNN-LSTM:
        # 1. Mean of windows within each recording
        recording_scores = (
            window_predictions.groupby(
                [
                    "participant_id",
                    "recording_id",
                    "label"
                ],
                as_index=False
            )["pd_probability"].mean()
        )

        # 2. Mean of recording scores per participant
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

        all_participants.append(participant_scores)

    final = pd.concat(
        all_participants, ignore_index=True
    )

    if len(final) != 89:
        raise ValueError("Expected 89 participants.")

    if final["participant_id"].duplicated().any():
        raise ValueError("Duplicate participants found.")

    return final

def calculate_metrics(df):

    actual = df["label"].to_numpy()
    predicted = df["prediction"].to_numpy()
    probabilities = df["pd_probability"].to_numpy()

    scores = {
        "Accuracy": accuracy_score(
            actual, predicted
        ) * 100,

        "Precision": precision_score(
            actual, predicted, zero_division=0
        ) * 100,

        "PD Recall": recall_score(
            actual, predicted, zero_division=0
        ) * 100,

        "F1-score": f1_score(
            actual, predicted, zero_division=0
        ) * 100,

        "ROC-AUC": roc_auc_score(
            actual, probabilities
        ) * 100,
    }

    matrix = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    return scores, matrix


def save_confusion(model_name, matrix):

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], ["Healthy", "Parkinson"])
    ax.set_yticks([0, 1], ["Healthy", "Parkinson"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_title(
        f"{model_name} - 89 Participant 5-Fold"
    )

    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                fontsize=16
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


def save_comparison(scores):

    metrics = [
        "Accuracy", "Precision",
        "PD Recall", "F1-score", "ROC-AUC"
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))

    for index, (name, values) in enumerate(
        scores.items()
    ):

        positions = x + (index - 1) * width

        heights = [values[m] for m in metrics]

        bars = ax.bar(
            positions, heights,
            width=width, label=name
        )

        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{value:.2f}",
                ha="center",
                fontsize=9
            )

    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")

    ax.set_title(
        "Parkinson's - 89 Participant "
        "5-Fold Model Comparison"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "fair_model_comparison.png",
        dpi=300
    )
    plt.close(fig)


def main():

    predictions = {}

    for model_name, (folder, model_class) in (
        MODEL_CLASSES.items()
    ):

        predictions[model_name] = (
            evaluate_saved_model(
                model_name, folder, model_class
            )
        )

    # Original CNN-LSTM already uses
    # recording-level participant averaging.
    cnn_lstm_file = (
        OLD_RESULTS /
        "pd_5fold_all_participant_predictions.csv"
    )

    predictions["CNN-LSTM"] = pd.read_csv(
        cnn_lstm_file,
        dtype={"participant_id": str}
    )

    # Check all participant labels match.
    reference = (
        predictions["CNN-LSTM"]
        .set_index("participant_id")["label"]
        .sort_index()
    )

    for name, df in predictions.items():

        current = (
            df.set_index("participant_id")["label"]
            .sort_index()
        )

        if not reference.equals(current):
            raise ValueError(
                f"Participant/label mismatch: {name}"
            )

    scores = {}

    for name, df in predictions.items():

        df.to_csv(
            OUTPUT / (
                name.lower().replace(" ", "_")
                .replace("-", "_")
                + "_participants.csv"
            ),
            index=False
        )

        values, matrix = calculate_metrics(df)
        scores[name] = values

        print(f"\n{name}")
        print(values)
        print(matrix)

        save_confusion(name, matrix)

    save_comparison(scores)

    summary = pd.DataFrame(scores).T
    summary.index.name = "Model"

    summary.to_csv(
        OUTPUT / "fair_model_comparison.csv"
    )

    print("\nCORRECTED COMPARISON COMPLETE")
    print("Saved in:", OUTPUT)


if __name__ == "__main__":
    main()
