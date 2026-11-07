from pathlib import Path
import zipfile
import re

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "parkinson" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "parkinson" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "1_10mSlow"
records = []

for zip_path in sorted(RAW_DIR.glob("*.zip")):

    zip_name = zip_path.name.lower()

    if "healthy" in zip_name:
        label = 0
        group = "Healthy"
    elif "patients" in zip_name:
        label = 1
        group = "Parkinson"
    else:
        continue

    with zipfile.ZipFile(zip_path, "r") as z:

        for name in z.namelist():

            if not name.lower().endswith(".csv"):
                continue

            if TASK.lower() not in [
                part.lower() for part in Path(name).parts
            ]:
                continue

            filename = Path(name).name

            match = re.match(
                r"^(\d{2})(?:-|[A-Za-z])",
                filename
            )

            if match is None:
                raise ValueError(
                    f"Unrecognised participant ID: {filename}"
                )

            participant_id = f"{group}_{match.group(1)}"

            records.append({
                "participant_id": participant_id,
                "group": group,
                "label": label,
                "task": TASK,
                "zip_path": str(zip_path),
                "csv_path": name,
            })


df = pd.DataFrame(records)

if len(df) != 89:
    raise ValueError(
        f"Expected 89 recordings; found {len(df)}"
    )

if df["participant_id"].nunique() != 89:
    raise ValueError(
        "Duplicate participant IDs detected. "
        "Inspect before splitting."
    )


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"],
)


splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df,
}

for name, split_df in splits.items():

    output_path = SPLIT_DIR / f"pd_{name}.csv"

    split_df.to_csv(output_path, index=False)

    print(f"\n{name.upper()}")
    print("-" * 35)
    print("Recordings:", len(split_df))
    print("Participants:", split_df["participant_id"].nunique())
    print("Healthy:", (split_df["label"] == 0).sum())
    print("Parkinson:", (split_df["label"] == 1).sum())
    print("Saved:", output_path)


# Verify no participant appears in multiple splits

train_ids = set(train_df["participant_id"])
val_ids = set(val_df["participant_id"])
test_ids = set(test_df["participant_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nParticipant leakage check: PASSED")
