from pathlib import Path
from collections import Counter, defaultdict
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "parkinson" / "raw"

zip_files = sorted(RAW_DIR.glob("*.zip"))

participants = defaultdict(set)
task_counts = defaultdict(Counter)
foot_counts = Counter()

print("PD DATASET PARTICIPANT AUDIT")
print("=" * 55)

for zip_path in zip_files:

    zip_name = zip_path.name.lower()

    if "healthy" in zip_name:
        group = "Healthy"
    elif "patients" in zip_name:
        group = "Parkinson"
    else:
        continue

    with zipfile.ZipFile(zip_path, "r") as z:

        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        for name in csv_files:

            parts = Path(name).parts

            filename = Path(name).name


            participant_id = filename.split("-")[0]

            participants[group].add(participant_id)


            task = Path(name).parent.name
            task_counts[group][task] += 1


            try:
                with z.open(name) as f:
                    df = pd.read_csv(
                        f,
                        usecols=["foot"]
                    )

                feet = set(
                    df["foot"]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .unique()
                )

                if {"left", "right"}.issubset(feet):
                    foot_counts["Both feet"] += 1
                elif "left" in feet:
                    foot_counts["Left only"] += 1
                elif "right" in feet:
                    foot_counts["Right only"] += 1
                else:
                    foot_counts["Unknown"] += 1

            except Exception:
                foot_counts["Read error"] += 1


print("\nPARTICIPANT COUNTS")
print("-" * 55)

for group in ["Healthy", "Parkinson"]:
    print(
        group,
        "unique IDs:",
        len(participants[group])
    )

    print(
        "Example IDs:",
        sorted(participants[group])[:10]
    )

print("\nFOOT AVAILABILITY")
print("-" * 55)

for key, value in foot_counts.items():
    print(key, ":", value)


print("\nCOMMON WALKING TASKS")
print("-" * 55)

healthy_tasks = set(task_counts["Healthy"])
pd_tasks = set(task_counts["Parkinson"])

common_tasks = healthy_tasks & pd_tasks

for task in sorted(common_tasks):
    print(
        task,
        "| Healthy:",
        task_counts["Healthy"][task],
        "| Parkinson:",
        task_counts["Parkinson"][task]
    )
