
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from pd_models import PDCNN1D, PDLSTM, PDCNNLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "parkinson" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "parkinson"
RESULT_DIR = PROJECT_ROOT / "results" / "parkinson"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

X = np.load(DATA_DIR / "pd_test_X.npy")
y = np.load(DATA_DIR / "pd_test_y.npy")
participants = np.load(
    DATA_DIR / "pd_test_participants.npy"
)

assert len(X) == len(y) == len(participants)
assert X.shape[1:] == (180, 12)

print("TEST DATA")
print("Windows:", len(X))
print("Participants:", len(np.unique(participants)))


models = {
    "1D CNN": (PDCNN1D, "pd_cnn_best.pt"),
    "LSTM": (PDLSTM, "pd_lstm_best.pt"),
    "CNN-LSTM": (PDCNNLSTM, "pd_cnn_lstm_best.pt")
}


def calculate_metrics(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    }, confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )


all_results = []

for model_name, (model_class, checkpoint) in models.items():

    print("\n" + "=" * 55)
    print("MODEL:", model_name)

    model = model_class().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_DIR / checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            ).float().to(DEVICE)

            output = model(batch)

            prob = torch.softmax(output, dim=1)[:, 1]

            probabilities.extend(
                prob.cpu().numpy().tolist()
            )

    probabilities = np.asarray(probabilities)

    window_metrics, window_cm = calculate_metrics(
        y, probabilities
    )

    window_df = pd.DataFrame({
        "participant_id": participants,
        "label": y,
        "pd_probability": probabilities
    })

    grouped = window_df.groupby("participant_id")

    participant_df = grouped.agg(
        label=("label", "first"),
        pd_probability=("pd_probability", "mean"),
        windows=("label", "size"),
        label_count=("label", "nunique")
    ).reset_index()

    if (participant_df["label_count"] != 1).any():
        raise ValueError(
            "Conflicting labels for the same participant"
        )

    participant_metrics, participant_cm = calculate_metrics(
        participant_df["label"].to_numpy(),
        participant_df["pd_probability"].to_numpy()
    )

    participant_df["prediction"] = (
        participant_df["pd_probability"] >= 0.5
    ).astype(int)

    model_key = model_name.lower().replace(" ", "_").replace("-", "_")

    participant_df.to_csv(
        RESULT_DIR / f"{model_key}_participants.csv",
        index=False
    )

    print("\nWINDOW-LEVEL METRICS")
    for name, value in window_metrics.items():
        print(f"{name}: {value:.4f}")

    print("Confusion matrix:")
    print(window_cm)

    print("\nPARTICIPANT-LEVEL METRICS")
    for name, value in participant_metrics.items():
        print(f"{name}: {value:.4f}")

    print("Confusion matrix:")
    print(participant_cm)

    for level, metrics in [
        ("Window", window_metrics),
        ("Participant", participant_metrics)
    ]:

        all_results.append({
            "Model": model_name,
            "Level": level,
            **metrics
        })


comparison = pd.DataFrame(all_results)

output_path = RESULT_DIR / "pd_model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print("FINAL PD MODEL COMPARISON")
print("=" * 55)

print(comparison.round(4).to_string(index=False))

print("\nSaved:", output_path)
