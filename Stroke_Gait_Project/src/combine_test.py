from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train_recordings.csv"
)

index_df = pd.read_csv(TRAIN_FILE)

print("COMBINE TEST")
print("--------------------------------")

print("Training recordings:", len(index_df))

# Take first training recording
row = index_df.iloc[0]

left_path = Path(row["left_foot_path"])
right_path = Path(row["right_foot_path"])

print("\nParticipant:")
print(row["participant_id"])

print("\nClass:")
print(row["class"])

print("\nRecording:")
print(row["recording_id"])

print("\nLeft file:")
print(left_path.name)

print("\nRight file:")
print(right_path.name)


left = pd.read_csv(left_path)
right = pd.read_csv(right_path)


print("\nOriginal Left shape:")
print(left.shape)

print("Original Right shape:")
print(right.shape)


left = left.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)

right = right.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


sensor_columns = [
    "ax", "ay", "az",
    "gx", "gy", "gz"
]

left = left[sensor_columns].copy()
right = right[sensor_columns].copy()

left = left.replace(
    [np.inf, -np.inf],
    np.nan
)

right = right.replace(
    [np.inf, -np.inf],
    np.nan
)

left = left.interpolate(
    method="linear",
    limit_direction="both"
)

right = right.interpolate(
    method="linear",
    limit_direction="both"
)


min_length = min(
    len(left),
    len(right)
)

left = left.iloc[:min_length].reset_index(drop=True)
right = right.iloc[:min_length].reset_index(drop=True)


left.columns = [
    "LF_ax",
    "LF_ay",
    "LF_az",
    "LF_gx",
    "LF_gy",
    "LF_gz"
]

right.columns = [
    "RF_ax",
    "RF_ay",
    "RF_az",
    "RF_gx",
    "RF_gy",
    "RF_gz"
]


combined = pd.concat(
    [left, right],
    axis=1
)


print("\n--------------------------------")
print("FINAL COMBINED DATA")
print("--------------------------------")

print("Shape:")
print(combined.shape)

print("\nColumns:")
print(combined.columns.tolist())

print("\nFirst 5 rows:")
print(combined.head())

print("\nMissing values:")
print(combined.isnull().sum())

