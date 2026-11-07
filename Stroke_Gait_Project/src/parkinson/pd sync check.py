from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = (
    PROJECT_ROOT / "data" / "parkinson"
    / "splits" / "pd_train.csv"
)

df_index = pd.read_csv(SPLIT_FILE)

row = df_index.iloc[0]

print("PARTICIPANT:", row["participant_id"])
print("GROUP:", row["group"])
print("TASK:", row["task"])

with zipfile.ZipFile(row["zip_path"], "r") as z:
    with z.open(row["csv_path"]) as file:
        df = pd.read_csv(file)

print("\nORIGINAL DATA")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

foot = df["foot"].astype(str).str.strip().str.lower()

left = df.loc[foot == "left"].copy()
right = df.loc[foot == "right"].copy()

print("\nFOOT DATA")
print("Left rows:", len(left))
print("Right rows:", len(right))

channels = [
    "acx", "acy", "acz",
    "gyrx", "gyry", "gyrz"
]

for name, sensor in [("LEFT", left), ("RIGHT", right)]:

    sensor["timestamp"] = pd.to_numeric(
        sensor["timestamp"], errors="coerce"
    )

    sensor = sensor.dropna(subset=["timestamp"])
    sensor = sensor.sort_values("timestamp")

    timestamps = sensor["timestamp"].to_numpy()
    differences = np.diff(timestamps)
    positive = differences[differences > 0]

    print(f"\n{name} SENSOR")
    print("Rows:", len(sensor))
    print("Unique timestamps:", sensor["timestamp"].nunique())

    if len(positive):
        median_dt = np.median(positive)
        print("Median interval:", median_dt, "seconds")
        print("Estimated frequency:", round(1 / median_dt, 2), "Hz")
    else:
        print("Unable to estimate sampling frequency")

    print("Missing IMU values:")
    print(sensor[channels].isna().sum().to_string())

if not left.empty and not right.empty:
    start = max(left["timestamp"].min(), right["timestamp"].min())
    end = min(left["timestamp"].max(), right["timestamp"].max())

    print("\nCOMMON TIME RANGE")
    print("Start:", start)
    print("End:", end)
    print("Overlap duration:", round(end - start, 2), "seconds")
else:
    print("\nERROR: One foot is missing")
