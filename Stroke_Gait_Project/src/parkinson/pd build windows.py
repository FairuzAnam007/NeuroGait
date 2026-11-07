from pathlib import Path

import numpy as np
import pandas as pd

from pd_preprocess import (
    load_pd_recording,
    synchronize_pd_feet
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "parkinson" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "parkinson" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 36
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
STEP_SIZE = WINDOW_SIZE // 2


def make_windows(signal):

    windows = []

    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):
        window = signal[start:start + WINDOW_SIZE]
        windows.append(window)

    if not windows:
        return np.empty(
            (0, WINDOW_SIZE, 12),
            dtype=np.float32
        )

    return np.stack(windows).astype(np.float32)


def process_split(split_name):

    index = pd.read_csv(
        SPLIT_DIR / f"pd_{split_name}.csv"
    )

    x_parts = []
    y_parts = []
    participant_parts = []

    failed = []

    for i, row in index.iterrows():

        try:
            df = load_pd_recording(
                row["zip_path"],
                row["csv_path"]
            )

            signal = synchronize_pd_feet(df)

            windows = make_windows(signal)

            if len(windows) == 0:
                raise ValueError("No valid windows")

            x_parts.append(windows)

            y_parts.append(
                np.full(
                    len(windows),
                    int(row["label"]),
                    dtype=np.int64
                )
            )

            participant_parts.append(
                np.array(
                    [row["participant_id"]] * len(windows),
                    dtype=str
                )
            )

        except Exception as error:
            failed.append(
                (row["participant_id"], str(error))
            )

        print(
            f"{split_name.upper()}: "
            f"{i + 1}/{len(index)}",
            end="\r"
        )

    if not x_parts:
        raise RuntimeError(
            f"No usable recordings in {split_name}"
        )

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    participants = np.concatenate(participant_parts, axis=0)

    np.save(
        OUTPUT_DIR / f"pd_{split_name}_X_raw.npy",
        X
    )

    np.save(
        OUTPUT_DIR / f"pd_{split_name}_y.npy",
        y
    )

    np.save(
        OUTPUT_DIR / f"pd_{split_name}_participants.npy",
        participants
    )

    print(f"\n{'=' * 45}")
    print("SPLIT:", split_name.upper())
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Healthy windows:", np.sum(y == 0))
    print("Parkinson windows:", np.sum(y == 1))

    print("Unique participants:", len(np.unique(participants)))
    print("Failed recordings:", len(failed))

    for item in failed[:10]:
        print("FAILED:", item)


for split in ["train", "val", "test"]:
    process_split(split)
