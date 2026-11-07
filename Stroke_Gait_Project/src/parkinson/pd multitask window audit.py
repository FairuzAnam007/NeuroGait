
from pathlib import Path
import numpy as np
import pandas as pd

from pd_preprocess import (
    load_pd_recording,
    synchronize_pd_feet
)

ROOT = Path(__file__).resolve().parents[2]

MULTI_DIR = (
    ROOT / "data" / "parkinson"
    / "splits" / "multitask"
)

INDEX_PATH = MULTI_DIR / "pd_multitask_final_index.csv"

SAMPLING_RATE = 36
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
STEP_SIZE = WINDOW_SIZE // 2
CHANNELS = 12

index = pd.read_csv(
    INDEX_PATH,
    dtype={"participant_id": str}
)

print("PD MULTI-TASK WINDOW AUDIT")
print("=" * 60)

print("Indexed recordings:", len(index))
print("Participants:", index["participant_id"].nunique())
print("Sampling rate:", SAMPLING_RATE, "Hz")
print("Window size:", WINDOW_SIZE, "samples")
print("Window step:", STEP_SIZE, "samples")

results = []

for i, row in index.iterrows():

    status = "OK"
    error = ""
    samples = 0
    windows = 0

    try:

        df = load_pd_recording(
            row["zip_path"],
            row["csv_path"]
        )

        signal = synchronize_pd_feet(df)

        signal = np.asarray(signal)

        if signal.ndim != 2:
            raise ValueError(
                f"Expected a 2D signal, got {signal.shape}"
            )

        if signal.shape[1] != CHANNELS:
            raise ValueError(
                f"Expected 12 channels, got {signal.shape}"
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                "Signal contains NaN or Inf values"
            )

        samples = len(signal)

        if samples < WINDOW_SIZE:
            status = "TOO_SHORT"
        else:
            windows = (
                1 + (samples - WINDOW_SIZE) // STEP_SIZE
            )

    except Exception as exc:
        status = "FAILED"
        error = str(exc)

    results.append({
        "participant_id": row["participant_id"],
        "label": row["label"],
        "split": row["split"],
        "task": row["task"],
        "zip_path": row["zip_path"],
        "csv_path": row["csv_path"],
        "samples": samples,
        "windows": windows,
        "status": status,
        "error": error
    })

    if (i + 1) % 50 == 0 or i + 1 == len(index):
        print(
            f"Processed {i + 1}/{len(index)} recordings"
        )


audit = pd.DataFrame(results)

output = MULTI_DIR / "pd_multitask_window_audit.csv"
audit.to_csv(output, index=False)

valid = audit[audit["status"] == "OK"]

print("\n" + "=" * 60)
print("FINAL AUDIT RESULTS")
print("=" * 60)

print("\nRECORDING STATUS")
print(audit["status"].value_counts().to_string())

print("\nTOTAL VALID RECORDINGS:", len(valid))

print(
    "VALID PARTICIPANTS:",
    valid["participant_id"].nunique()
)

print(
    "TOTAL WINDOWS:",
    int(valid["windows"].sum())
)

print("\nWINDOWS BY TASK")
print(
    valid.groupby("task")["windows"]
    .sum()
    .to_string()
)

print("\nWINDOWS BY ORIGINAL SPLIT")
print(
    valid.groupby("split")["windows"]
    .sum()
    .to_string()
)

print("\nVALID PARTICIPANTS BY CLASS")
print(
    valid.groupby("label")["participant_id"]
    .nunique()
    .to_string()
)

problems = audit[audit["status"] != "OK"]

if not problems.empty:
    print("\nRECORDINGS REQUIRING REVIEW")
    print(
        problems[
            ["participant_id", "task", "status", "error"]
        ].to_string(index=False)
    )

print("\nSaved:", output)
