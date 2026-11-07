from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT / "results" / "parkinson" / "cv_multitask"
)

file_path = (
    RESULT_DIR /
    "pd_5fold_all_window_predictions.csv"
)

df = pd.read_csv(
    file_path,
    dtype={"participant_id": str}
)

results = []

print("PARKINSON TASK-WISE EVALUATION")
print("=" * 65)

for task in sorted(df["task"].unique()):

    task_data = df[df["task"] == task].copy()

    recording_scores = (
        task_data.groupby(
            [
                "participant_id",
                "recording_id",
                "label"
            ],
            as_index=False
        )["pd_probability"].mean()
    )

    participants = (
        recording_scores.groupby(
            ["participant_id", "label"],
            as_index=False
        )["pd_probability"].mean()
    )

    participants["prediction"] = (
        participants["pd_probability"] >= 0.5
    ).astype(int)

    y_true = participants["label"]
    y_pred = participants["prediction"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    results.append({
        "task": task,
        "participants": len(participants),
        "windows": len(task_data),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "pd_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })


result = pd.DataFrame(results)

print("\nTASK COMPARISON")
print("=" * 65)

display = result.copy()

for column in [
    "accuracy",
    "precision",
    "pd_recall",
    "f1_score"
]:
    display[column] = (
        display[column] * 100
    ).round(2)

print(display.to_string(index=False))

output = RESULT_DIR / "pd_taskwise_metrics.csv"

result.to_csv(output, index=False)

print("\nSaved:", output)
print("\nSTEP 24A COMPLETED")
