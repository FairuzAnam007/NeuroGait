from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from pd_models import PDCNNLSTM


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "parkinson" / "processed"
MODEL_DIR = ROOT / "models" / "parkinson"
RESULT_DIR = ROOT / "results" / "parkinson"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32


model = PDCNNLSTM().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "pd_cnn_lstm_best.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


def predict_participants(split):

    X = np.load(DATA_DIR / f"pd_{split}_X.npy")
    y = np.load(DATA_DIR / f"pd_{split}_y.npy")

    participants = np.load(
        DATA_DIR / f"pd_{split}_participants.npy"
    )

    probabilities = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            ).float().to(DEVICE)

            output = model(batch)

            prob = torch.softmax(
                output, dim=1
            )[:, 1]

            probabilities.extend(
                prob.cpu().numpy().tolist()
            )

    df = pd.DataFrame({
        "participant": participants,
        "label": y,
        "probability": probabilities
    })

    if (df.groupby("participant")["label"].nunique() != 1).any():
        raise ValueError("Conflicting participant labels")

    result = df.groupby("participant").agg(
        label=("label", "first"),
        probability=("probability", "mean")
    ).reset_index()

    return result


def evaluate(df, threshold):

    actual = df["label"].to_numpy()

    predicted = (
        df["probability"].to_numpy() >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "cm": confusion_matrix(
            actual, predicted, labels=[0, 1]
        )
    }


validation = predict_participants("val")

thresholds = np.arange(0.10, 0.91, 0.05)

rows = []

for threshold in thresholds:

    result = evaluate(validation, threshold)

    rows.append({
        "threshold": round(float(threshold), 2),
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"]
    })

table = pd.DataFrame(rows)

table["distance"] = abs(table["threshold"] - 0.5)

best = table.sort_values(
    ["f1", "distance"],
    ascending=[False, True]
).iloc[0]

best_threshold = float(best["threshold"])

print("VALIDATION THRESHOLD ANALYSIS")
print(table.drop(columns="distance").round(4).to_string(index=False))

print("\nSelected threshold:", best_threshold)

test = predict_participants("test")

print("\nUNSEEN TEST COMPARISON")

for threshold in [0.5, best_threshold]:

    result = evaluate(test, threshold)

    print("\nThreshold:", threshold)
    print("Accuracy:", round(result["accuracy"] * 100, 2), "%")
    print("Precision:", round(result["precision"] * 100, 2), "%")
    print("PD Recall:", round(result["recall"] * 100, 2), "%")
    print("F1:", round(result["f1"] * 100, 2), "%")
    print("Confusion matrix:")
    print(result["cm"])


table.drop(columns="distance").to_csv(
    RESULT_DIR / "pd_threshold_validation.csv",
    index=False
)

np.save(
    MODEL_DIR / "pd_selected_threshold.npy",
    np.array(best_threshold)
)
