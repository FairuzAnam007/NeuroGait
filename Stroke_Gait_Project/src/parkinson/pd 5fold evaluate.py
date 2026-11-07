
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT / "results" / "parkinson" / "cv_multitask"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(df, name):
    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["prediction"].astype(int).to_numpy()
    y_prob = df["pd_probability"].to_numpy()

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Total samples:", len(df))
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"PD Recall: {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nCorrect Healthy:", cm[0, 0])
    print("Healthy predicted PD:", cm[0, 1])
    print("PD predicted Healthy:", cm[1, 0])
    print("Correct PD:", cm[1, 1])

    # Save confusion matrix figure
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Parkinson"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    ax.set_title(name)

    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    fig.savefig(
        RESULT_DIR / f"{filename}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "evaluation": name,
        "samples": len(df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }

all_windows = []
all_participants = []

for fold in range(1, 6):

    window_file = (
        RESULT_DIR / f"fold_{fold}_windows.csv"
    )

    participant_file = (
        RESULT_DIR / f"fold_{fold}_participants.csv"
    )

    windows = pd.read_csv(
        window_file,
        dtype={"participant_id": str}
    )

    participants = pd.read_csv(
        participant_file,
        dtype={"participant_id": str}
    )

    assert len(participants) == (17 if fold == 5 else 18)

    assert (participants["fold"] == fold).all()

    all_windows.append(windows)
    all_participants.append(participants)


windows_df = pd.concat(
    all_windows, ignore_index=True
)

participants_df = pd.concat(
    all_participants, ignore_index=True
)

# Verify out-of-fold coverage
assert len(windows_df) == 5072
assert len(participants_df) == 89

assert participants_df["participant_id"].is_unique

assert windows_df["window_index"].is_unique
assert windows_df["window_index"].between(0, 5071).all()

assert windows_df["participant_id"].nunique() == 89

assert participants_df["label"].value_counts().to_dict() == {
    0: 45,
    1: 44
}

assert (
    windows_df.groupby("participant_id")["label"]
    .nunique() == 1
).all()

assert windows_df["pd_probability"].between(0, 1).all()
assert participants_df["pd_probability"].between(0, 1).all()

print("PARKINSON MULTI-TASK 5-FOLD EVALUATION")
print("Out-of-fold coverage check: PASSED")

window_results = calculate_metrics(
    windows_df,
    "Window Level 5 Fold"
)

participant_results = calculate_metrics(
    participants_df,
    "Participant Level 5 Fold"
)

summary = pd.DataFrame([
    window_results,
    participant_results
])

summary.to_csv(
    RESULT_DIR / "pd_5fold_final_metrics.csv",
    index=False
)

windows_df.to_csv(
    RESULT_DIR / "pd_5fold_all_window_predictions.csv",
    index=False
)

participants_df.to_csv(
    RESULT_DIR / "pd_5fold_all_participant_predictions.csv",
    index=False
)

print("\n" + "=" * 55)
print("FINAL COMPARISON")
print("=" * 55)

print(
    summary[
        [
            "evaluation",
            "samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ].round(4).to_string(index=False)
)

print("\nResults saved:", RESULT_DIR)

