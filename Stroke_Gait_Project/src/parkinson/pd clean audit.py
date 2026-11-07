from pathlib import Path
from collections import defaultdict, Counter
import zipfile
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "parkinson" / "raw"

TASK = "1_10mSlow"

participants = defaultdict(set)
recordings = Counter()
invalid_ids = []
duplicate_records = []
seen_records = set()

print("PARKINSON DATASET CLEAN AUDIT")
print("=" * 55)

for zip_path in sorted(RAW_DIR.glob("*.zip")):

    zip_name = zip_path.name.lower()

    if "healthy" in zip_name:
        group = "Healthy"
    elif "patients" in zip_name:
        group = "Parkinson"
    else:
        continue

    with zipfile.ZipFile(zip_path, "r") as z:

        for name in z.namelist():

            if not name.lower().endswith(".csv"):
                continue

            # Select the common walking task
            path_parts = Path(name).parts

            if TASK.lower() not in [
                part.lower() for part in path_parts
            ]:
                continue

            filename = Path(name).name


            match = re.match(
                r"^(\d{2})(?:-|[A-Za-z])",
                filename
            )

            if not match:
                invalid_ids.append((group, filename))
                continue

            participant_id = match.group(1)
            unique_id = f"{group}_{participant_id}"

            participants[group].add(unique_id)
            recordings[group] += 1

            record_key = (group, filename.lower())

            if record_key in seen_records:
                duplicate_records.append(record_key)

            seen_records.add(record_key)


print("\nTASK:", TASK)

for group in ["Healthy", "Parkinson"]:

    print("\nGROUP:", group)
    print("Recordings:", recordings[group])
    print("Unique participants:", len(participants[group]))
    print("Participant IDs:", sorted(participants[group]))

print("\nInvalid participant IDs:", len(invalid_ids))

for item in invalid_ids[:10]:
    print(item)

print("\nRepeated recording names:", len(duplicate_records))

for item in duplicate_records[:10]:
    print(item)
