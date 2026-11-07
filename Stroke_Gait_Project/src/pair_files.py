from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

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


for file in csv_files:

    sensor = get_sensor_type(file.name)

    if sensor is None:
        continue

    key = get_recording_key(file.name)

    recordings[key][sensor] = file


paired_recordings = []

missing_left = []
missing_right = []


for key, sensors in recordings.items():

    has_left = "leftfoot" in sensors
    has_right = "rightfoot" in sensors

    if has_left and has_right:

        paired_recordings.append(
            {
                "recording": key,
                "leftfoot": sensors["leftfoot"],
                "rightfoot": sensors["rightfoot"],
                "lowback": sensors.get("lowback")
            }
        )

    else:

        if not has_left:
            missing_left.append(key)

        if not has_right:
            missing_right.append(key)


print("PAIRING SUMMARY")
print("--------------------------------")

print("Total CSV files:", len(csv_files))
print("Total recording groups:", len(recordings))

print(
    "Valid Left + Right pairs:",
    len(paired_recordings)
)

print(
    "Recordings missing Left Foot:",
    len(missing_left)
)

print(
    "Recordings missing Right Foot:",
    len(missing_right)
)


print("\nFIRST 10 VALID PAIRS")
print("--------------------------------")

for item in paired_recordings[:10]:

    print("\nRecording:", item["recording"])
    print(
        "Left :",
        item["leftfoot"].name
    )
    print(
        "Right:",
        item["rightfoot"].name
    )

    if item["lowback"]:
        print(
            "Back :",
            item["lowback"].name
        )
    else:
        print("Back : Not available")
