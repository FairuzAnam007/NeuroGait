from pathlib import Path
import pandas as pd
import numpy as np

from preprocess import synchronize_feet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train_recordings.csv"
)


SAMPLING_RATE = 104

WINDOW_SECONDS = 5

WINDOW_SIZE = (
    SAMPLING_RATE
    * WINDOW_SECONDS
)

OVERLAP = 0.50

STEP_SIZE = int(
    WINDOW_SIZE
    * (1 - OVERLAP)
)


print("WINDOW TEST")
print("--------------------------------")

print("Sampling rate:", SAMPLING_RATE)
print("Window seconds:", WINDOW_SECONDS)
print("Window size:", WINDOW_SIZE)
print("Overlap:", OVERLAP)
print("Step size:", STEP_SIZE)


index_df = pd.read_csv(
    TRAIN_FILE
)

row = index_df.iloc[0]

print("\nParticipant:")
print(row["participant_id"])

print("Class:")
print(row["class"])

print("Recording:")
print(row["recording_id"])


combined = synchronize_feet(
    Path(row["left_foot_path"]),
    Path(row["right_foot_path"])
)

print("\nSynchronised shape:")
print(combined.shape)


data = combined.to_numpy(
    dtype=np.float32
)

windows = []

for start in range(
    0,
    len(data) - WINDOW_SIZE + 1,
    STEP_SIZE
):

    end = start + WINDOW_SIZE

    window = data[
        start:end
    ]

    windows.append(
        window
    )


windows = np.array(
    windows,
    dtype=np.float32
)


print("\n--------------------------------")
print("WINDOW OUTPUT")
print("--------------------------------")

print("Total windows:")
print(len(windows))

print("\nWindow array shape:")
print(windows.shape)


if len(windows) > 0:

    print("\nSingle window shape:")
    print(
        windows[0].shape
    )


print("\nExpected single window:")
print("(520, 12)")


