from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parkinson"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

models = {
    "1D CNN": "1d_cnn_participants.csv",
    "LSTM": "lstm_participants.csv",
    "CNN-LSTM": "cnn_lstm_participants.csv"
}


for model_name, filename in models.items():

    file_path = RESULT_DIR / filename

    df = pd.read_csv(file_path)

    y_true = df["label"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Healthy",
            "Parkinson's"
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} - Participant-Level Confusion Matrix"
    )

    plt.tight_layout()

    save_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    save_path = RESULT_DIR / save_name

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nMODEL:", model_name)
    print("Confusion matrix:")
    print(cm)
    print("Saved:", save_path)
