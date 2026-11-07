from pathlib import Path
from collections import defaultdict
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "recording_index.csv"

csv_files = list(RAW_DIR.rglob("*.csv"))

recordings = defaultdict(dict)


def get_sensor_type(filename):
    name = filename.lower()

    if name.endswith("_leftfoot.csv"):
        return "leftfoot"

    if name.endswith("_rightfoot.csv"):
        return "rightfoot"

    if name.endswith("_lowback.csv"):
        return "lowback"

    if name.endswith("_lowerback.csv"):
        return "lowback"

    return None


def get_recording_key(filename):
    stem = Path(filename).stem

    endings = [
        "_leftfoot",
        "_rightfoot",
        "_lowback",
        "_lowerback"
    ]

    for ending in endings:
        if stem.lower().endswith(ending):
            return stem[:-len(ending)]

    return stem


def get_class(recording_key):
    name = recording_key.lower()

    if "_healthy_" in name:
        return "Healthy"

    if "_stroke_" in name:
        return "Stroke"

    return "Unknown"


def get_raw_id(recording_key):
    return recording_key.split("_")[0]


def get_base_id(raw_id):


    match = re.match(r"([A-Za-z]+\d+)", raw_id)

    if match:
        return match.group(1)

    return raw_id


for file in csv_files:

    sensor = get_sensor_type(file.name)

    if sensor is None:
        continue

    key = get_recording_key(file.name)

    recordings[key][sensor] = file


rows = []


for recording_id, sensors in recordings.items():

    if "leftfoot" not in sensors:
        continue

    if "rightfoot" not in sensors:
        continue

    raw_id = get_raw_id(recording_id)
    base_id = get_base_id(raw_id)

    rows.append(
        {
            "participant_id": base_id,
            "raw_id": raw_id,
            "class": get_class(recording_id),
            "recording_id": recording_id,
            "left_foot_path": str(sensors["leftfoot"]),
            "right_foot_path": str(sensors["rightfoot"]),
            "low_back_path": str(sensors.get("lowback", ""))
        }
    )


df = pd.DataFrame(rows)

df = df.sort_values(
    ["class", "participant_id", "recording_id"]
).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("RECORDING INDEX CREATED")
print("--------------------------------")

print("Saved to:")
print(OUTPUT_FILE)

print("\nTotal usable recordings:")
print(len(df))

print("\nClass counts:")
print(df["class"].value_counts())

print("\nUnique base participant IDs:")
print(df["participant_id"].nunique())

print("\nFirst 10 rows:")
print(
    df[
        [
            "participant_id",
            "raw_id",
            "class",
            "recording_id"
        ]
    ].head(10)
)
