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

OUT_DIR = (
    ROOT / "data" / "parkinson"
    / "processed" / "multitask"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 180
STEP_SIZE = 90
CHANNELS = 12

audit = pd.read_csv(
    MULTI_DIR / "pd_multitask_window_audit.csv",
    dtype={"participant_id": str}
)

valid = audit[audit["status"] == "OK"].copy()

print("BUILDING MULTI-TASK WINDOWS")
print("=" * 55)
print("Valid recordings:", len(valid))

X_all = []
metadata = []
failed = []

for recording_id, (_, row) in enumerate(
    valid.iterrows(), start=1
):

    try:
        df = load_pd_recording(
            row["zip_path"],
            row["csv_path"]
        )

        signal = np.asarray(
            synchronize_pd_feet(df),
            dtype=np.float32
        )

        if (
            signal.ndim != 2
            or signal.shape[1] != CHANNELS
            or not np.isfinite(signal).all()
        ):
            raise ValueError("Invalid IMU signal")

        count = 0

        for start in range(
            0,
            len(signal) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):
            window = signal[
                start:start + WINDOW_SIZE
            ]

            X_all.append(window)

            metadata.append({
                "participant_id": row["participant_id"],
                "label": int(row["label"]),
                "original_split": row["split"],
                "task": row["task"],
                "recording_id": recording_id,
                "window_start": start,
                "zip_path": row["zip_path"],
                "csv_path": row["csv_path"]
            })

            count += 1

        if count != int(row["windows"]):
            raise ValueError(
                f"Window count mismatch: "
                f"{count} vs {row['windows']}"
            )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to process recording: "
            f"{row['csv_path']}"
        ) from exc

    if recording_id % 50 == 0:
        print(
            f"Processed {recording_id}/{len(valid)}"
        )


X = np.stack(X_all).astype(np.float32)
meta = pd.DataFrame(metadata)

assert X.ndim == 3
assert X.shape[1:] == (180, 12)
assert np.isfinite(X).all()

assert meta["participant_id"].nunique() == 89

assert len(X) == len(meta)

expected_windows = int(valid["windows"].sum())

assert len(X) == expected_windows, (
    f"Expected {expected_windows}, got {len(X)}"
)

np.save(
    OUT_DIR / "pd_multitask_X_raw.npy",
    X
)

meta.to_csv(
    OUT_DIR / "pd_multitask_metadata.csv",
    index=False
)

print("\n" + "=" * 55)
print("FINAL DATASET")
print("=" * 55)

print("Windows:", len(X))
print("Shape:", X.shape)
print("Participants:",
      meta["participant_id"].nunique())

print("\nWindows by label:")
print(meta["label"].value_counts().sort_index())

print("\nWindows by original split:")
print(meta["original_split"].value_counts())

print("\nSaved:")
print(OUT_DIR / "pd_multitask_X_raw.npy")
print(OUT_DIR / "pd_multitask_metadata.csv")
