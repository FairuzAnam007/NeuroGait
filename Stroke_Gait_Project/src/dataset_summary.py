from pathlib import Path
from collections import Counter, defaultdict
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

print("SCANNING DATASET...")
print("------------------------------")

csv_files = list(RAW_DIR.rglob("*.csv"))

print("Total CSV files:", len(csv_files))

class_count = Counter()
location_count = Counter()

participant_files = defaultdict(list)


def detect_class(filename):
    name = filename.lower()

    if "healthy" in name:
        return "Healthy"

    if "stroke" in name:
        return "Stroke"

    return "Unknown"


def detect_location(filename):
    name = filename.lower()

    if "leftfoot" in name or "left_foot" in name:
        return "Left Foot"

    if "rightfoot" in name or "right_foot" in name:
        return "Right Foot"

    if "lowerback" in name or "lower_back" in name:
        return "Lower Back"

    return "Unknown"


def extract_participant_id(filename):
    name = Path(filename).stem


    participant_id = name.split("_")[0]

    return participant_id


for file in csv_files:

    cls = detect_class(file.name)
    loc = detect_location(file.name)
    pid = extract_participant_id(file.name)

    class_count[cls] += 1
    location_count[loc] += 1
    participant_files[pid].append(file)


print("\nCLASS COUNTS")
print("------------------------------")
for key, value in class_count.items():
    print(key, ":", value)


print("\nSENSOR LOCATION COUNTS")
print("------------------------------")
for key, value in location_count.items():
    print(key, ":", value)


print("\nUNIQUE PARTICIPANT IDs")
print("------------------------------")
print("Total unique IDs:", len(participant_files))


print("\nFIRST 20 PARTICIPANT IDs")
print("------------------------------")
for pid in list(participant_files.keys())[:20]:
    print(pid, "->", len(participant_files[pid]), "files")

