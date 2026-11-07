from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from pd_models import PDCNNLSTM
from pd_preprocess import (
    load_pd_recording,
    synchronize_pd_feet
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT / "models" / "parkinson" / "final_multitask"
)

INDEX_PATH = (
    ROOT / "data" / "parkinson"
    / "splits" / "multitask"
    / "pd_multitask_final_index.csv"
)

DEVICE = torch.device("cpu")
WINDOW_SIZE = 180
STEP_SIZE = 90
THRESHOLD = 0.50


def predict_signal(df):

    signal = np.asarray(
        synchronize_pd_feet(df),
        dtype=np.float32
    )

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("Expected 12 IMU channels")

    if not np.isfinite(signal).all():
        raise ValueError("Invalid IMU values")

    windows = []

    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):
        windows.append(
            signal[start:start + WINDOW_SIZE]
        )

    if not windows:
        raise ValueError(
            "Recording has less than 5 seconds "
            "of usable bilateral IMU data"
        )

    X = np.stack(windows)

    mean = np.load(
        MODEL_DIR / "pd_final_mean.npy"
    )

    std = np.load(
        MODEL_DIR / "pd_final_std.npy"
    )

    X = (X - mean) / std

    model = PDCNNLSTM().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "pd_final_cnn_lstm.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for start in range(0, len(X), 32):

            batch = torch.from_numpy(
                X[start:start + 32]
            ).float().to(DEVICE)

            outputs = model(batch)

            pd_prob = torch.softmax(
                outputs, dim=1
            )[:, 1]

            probabilities.extend(
                pd_prob.cpu().numpy().tolist()
            )

    average_probability = float(
        np.mean(probabilities)
    )

    prediction = (
        "Parkinson's-associated gait"
        if average_probability >= THRESHOLD
        else "Healthy-associated gait"
    )

    return {
        "windows": len(X),
        "pd_probability": average_probability,
        "prediction": prediction
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        help="Path to an extracted PD-format CSV"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use one indexed training-data recording"
    )

    args = parser.parse_args()

    if not args.csv and not args.demo:
        args.demo = True

    if args.demo:

        index = pd.read_csv(INDEX_PATH)

        row = index[
            index["task"] == "1_10mSlow"
        ].iloc[0]

        df = load_pd_recording(
            row["zip_path"],
            row["csv_path"]
        )

        source = row["csv_path"]

        print(
            "DEMO ONLY: This recording was included "
            "in final-model training."
        )

    elif args.csv:

        source = str(Path(args.csv).resolve())
        df = pd.read_csv(source)

    else:
        parser.error(
            "Provide --csv FILE or --demo"
        )

    result = predict_signal(df)

    print("\nFINAL PARKINSON PREDICTION")
    print("=" * 50)

    print("Recording:", source)
    print("Windows:", result["windows"])

    print(
        "PD model score:",
        f"{result['pd_probability'] * 100:.2f}%"
    )

    print("Prediction:", result["prediction"])
    print("Threshold:", THRESHOLD)

    print(
        "\nResearch screening output only. "
        "Not a medical diagnosis."
    )


if __name__ == "__main__":
    main()
