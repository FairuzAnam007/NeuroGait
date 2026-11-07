from pathlib import Path

import numpy as np
import torch
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

from models import CNNLSTM


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_lstm_best.pt"
RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:")
print(device)


X_test = np.load(
    DATA_DIR / "test_X.npy"
)

y_test = np.load(
    DATA_DIR / "test_y.npy"
)

participants = np.load(
    DATA_DIR / "test_participants.npy"
)


print("\nTEST DATA")
print("------------------------------")

print("X shape:", X_test.shape)
print("y shape:", y_test.shape)

print(
    "Unique participants:",
    len(np.unique(participants))
)


model = CNNLSTM().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


batch_size = 32

all_probabilities = []


with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        batch_size
    ):

        end = start + batch_size

        X_batch = torch.tensor(
            X_test[start:end],
            dtype=torch.float32
        ).to(device)

        outputs = model(
            X_batch
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        stroke_prob = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        all_probabilities.extend(
            stroke_prob
        )


all_probabilities = np.array(
    all_probabilities
)


participant_results = []

unique_participants = np.unique(
    participants
)


for participant_id in unique_participants:

    mask = (
        participants == participant_id
    )

    participant_probs = (
        all_probabilities[mask]
    )

    participant_labels = (
        y_test[mask]
    )

    actual_label = int(
        participant_labels[0]
    )

    mean_stroke_probability = float(
        participant_probs.mean()
    )

    predicted_label = (
        1
        if mean_stroke_probability >= 0.50
        else 0
    )

    participant_results.append(
        {
            "participant_id": participant_id,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "stroke_probability": mean_stroke_probability,
            "num_windows": int(mask.sum())
        }
    )


participant_true = np.array(
    [
        item["actual_label"]
        for item in participant_results
    ]
)

participant_pred = np.array(
    [
        item["predicted_label"]
        for item in participant_results
    ]
)

participant_prob = np.array(
    [
        item["stroke_probability"]
        for item in participant_results
    ]
)


accuracy = accuracy_score(
    participant_true,
    participant_pred
)

precision = precision_score(
    participant_true,
    participant_pred
)

recall = recall_score(
    participant_true,
    participant_pred
)

f1 = f1_score(
    participant_true,
    participant_pred
)

auc = roc_auc_score(
    participant_true,
    participant_prob
)


print("\nPARTICIPANT-LEVEL RESULTS")
print("--------------------------------")

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1-score:  {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:   {auc:.4f}"
)


print("\nINDIVIDUAL PARTICIPANTS")
print("--------------------------------")

for item in participant_results:

    actual_name = (
        "Stroke"
        if item["actual_label"] == 1
        else "Healthy"
    )

    predicted_name = (
        "Stroke"
        if item["predicted_label"] == 1
        else "Healthy"
    )

    print(
        item["participant_id"],
        "| Actual:",
        actual_name,
        "| Predicted:",
        predicted_name,
        "| Stroke probability:",
        f'{item["stroke_probability"] * 100:.2f}%',
        "| Windows:",
        item["num_windows"]
    )


cm = confusion_matrix(
    participant_true,
    participant_pred
)

print("\nParticipant Confusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Stroke"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "CNN-LSTM Participant-Level Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "participant_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


import pandas as pd

participant_df = pd.DataFrame(
    participant_results
)

participant_df["actual_class"] = (
    participant_df["actual_label"]
    .map(
        {
            0: "Healthy",
            1: "Stroke"
        }
    )
)

participant_df["predicted_class"] = (
    participant_df["predicted_label"]
    .map(
        {
            0: "Healthy",
            1: "Stroke"
        }
    )
)

participant_df[
    "stroke_probability_percent"
] = (
    participant_df[
        "stroke_probability"
    ]
    * 100
)

participant_df.to_csv(
    RESULT_DIR
    / "participant_predictions.csv",
    index=False
)


print("\nSaved:")
print(
    RESULT_DIR
    / "participant_predictions.csv"
)

print(
    RESULT_DIR
    / "participant_confusion_matrix.png"
)

