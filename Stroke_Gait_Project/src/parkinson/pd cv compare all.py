from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "parkinson"

NEW_RESULTS = RESULTS / "cv_compare_89"
OLD_RESULTS = RESULTS / "cv_multitask"

OUTPUT = NEW_RESULTS / "final_comparison_graphs"
OUTPUT.mkdir(parents=True, exist_ok=True)

CNN_FILE = (
    NEW_RESULTS / "cnn" /
    "all_participant_predictions.csv"
)

LSTM_FILE = (
    NEW_RESULTS / "lstm" /
    "all_participant_predictions.csv"
)

CNN_LSTM_FILE = (
    OLD_RESULTS /
    "pd_5fold_all_participant_predictions.csv"
)

if not CNN_LSTM_FILE.exists():
    candidates = [
        p for p in OLD_RESULTS.glob("*.csv")
        if "participant" in p.name.lower()
        and "prediction" in p.name.lower()
    ]

    if len(candidates) == 1:
        CNN_LSTM_FILE = candidates[0]

def find_column(df, names):

    def clean(text):
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    available = {
        clean(col): col
        for col in df.columns
    }

    for name in names:
        key = clean(name)

        if key in available:
            return available[key]

    raise ValueError(
        f"Cannot find column among {names}\n"
        f"Available columns: {list(df.columns)}"
    )


def to_binary_label(value):

    text = str(value).strip().lower()

    if text in [
        "0", "0.0", "healthy",
        "normal", "control", "hc"
    ]:
        return 0

    if text in [
        "1", "1.0", "pd", "parkinson",
        "parkinsons", "parkinson's"
    ]:
        return 1

    raise ValueError(f"Unknown label: {value}")

def load_predictions(file_path, model_name):

    if not file_path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found:\n"
            f"{file_path}"
        )

    df = pd.read_csv(file_path)

    actual_col = find_column(
        df,
        [
            "true_label", "actual_label",
            "label", "y_true", "true_class",
            "actual", "target_label"
        ]
    )

    predicted_col = find_column(
        df,
        [
            "prediction", "predicted_label",
            "pred_label", "y_pred",
            "predicted_class", "predicted"
        ]
    )

    probability_col = find_column(
        df,
        [
            "pd_probability",
            "pd_prob", "mean_pd_probability",
            "mean_pd_prob", "mean_probability",
            "avg_probability", "average_probability",
            "probability", "mean_prob",
            "avg_prob", "pd_score",
            "prob_pd", "score"
        ]
    )

    id_col = find_column(
        df,
        [
            "participant_key", "participant_id",
            "participant", "subject_id",
            "subject", "patient_id", "pid"
        ]
    )

    result = pd.DataFrame({
        "participant_id": df[id_col].astype(str).str.strip(),
        "actual": df[actual_col].map(to_binary_label),
        "predicted": df[predicted_col].map(to_binary_label),
        "probability": pd.to_numeric(
            df[probability_col], errors="raise"
        )
    })

    if result["participant_id"].duplicated().any():
        raise ValueError(
            f"Duplicate participant IDs in {model_name}"
        )

    if len(result) != 89:
        raise ValueError(
            f"{model_name}: expected 89 participants, "
            f"found {len(result)}"
        )

    if not result["probability"].between(0, 1).all():
        raise ValueError(
            f"{model_name}: PD scores must be between 0 and 1"
        )

    print(f"{model_name}: {len(result)} participants loaded")
    print(f"Source: {file_path}")

    return result

def calculate_results(df):

    actual = df["actual"].to_numpy()
    predicted = df["predicted"].to_numpy()
    probability = df["probability"].to_numpy()

    scores = {
        "Accuracy": accuracy_score(actual, predicted) * 100,
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
            actual, probability
        ) * 100
    }

    matrix = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    return scores, matrix

def plot_comparison(all_scores):

    metrics = [
        "Accuracy", "Precision",
        "PD Recall", "F1-score", "ROC-AUC"
    ]

    models = list(all_scores.keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, model in enumerate(models):

        values = [
            all_scores[model][metric]
            for metric in metrics
        ]

        positions = x + (i - 1) * width

        bars = ax.bar(
            positions, values,
            width=width, label=model
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.2f}",
                ha="center",
                fontsize=9
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)

    ax.set_title(
        "Parkinson's Classification - "
        "89 Participant 5-Fold Comparison"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    file = OUTPUT / "all_models_comparison.png"

    plt.savefig(file, dpi=300)
    plt.close()

    print("Saved:", file)

def plot_confusion_matrix(model_name, matrix):

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=45
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Healthy", "Parkinson"])
    ax.set_yticklabels(["Healthy", "Parkinson"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_title(
        f"{model_name} - Participant-Level\n"
        f"5-Fold Confusion Matrix (N=89)"
    )

    for i in range(2):
        for j in range(2):
            value = matrix[i, j]

            ax.text(
                j, i, str(value),
                ha="center",
                va="center",
                fontsize=18,
                color=(
                    "white" if value > 22
                    else "black"
                )
            )

    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    file = OUTPUT / filename

    plt.savefig(file, dpi=300)
    plt.close()

    print("Saved:", file)

def main():

    files = {
        "1D CNN": CNN_FILE,
        "LSTM": LSTM_FILE,
        "CNN-LSTM": CNN_LSTM_FILE
    }

    predictions = {}

    for model_name, file_path in files.items():
        predictions[model_name] = load_predictions(
            file_path, model_name
        )

    reference = predictions["1D CNN"].set_index(
        "participant_id"
    )["actual"].sort_index()

    for name, df in predictions.items():

        current = df.set_index(
            "participant_id"
        )["actual"].sort_index()

        if not reference.equals(current):
            raise ValueError(
                f"{name} does not have the same "
                "participants and labels."
            )

    all_scores = {}

    for model_name, df in predictions.items():

        scores, matrix = calculate_results(df)

        all_scores[model_name] = scores

        print("\nMODEL:", model_name)
        print("Metrics:", scores)
        print("Confusion Matrix:\n", matrix)

        plot_confusion_matrix(
            model_name, matrix
        )

    plot_comparison(all_scores)

    comparison = pd.DataFrame(
        all_scores
    ).T

    comparison.index.name = "Model"

    comparison.to_csv(
        OUTPUT / "final_model_comparison.csv"
    )

    print("\n" + "=" * 50)
    print("ALL GRAPHICAL RESULTS GENERATED")
    print("=" * 50)
    print("Folder:", OUTPUT)


if __name__ == "__main__":
    main()
