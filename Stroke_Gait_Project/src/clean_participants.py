from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recording_index.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recording_index_clean.csv"
)

df = pd.read_csv(INDEX_FILE)


ID_CORRECTIONS = {
    "S5223": "S6669",
    "S2706": "S9317",
    "S5649": "S6567",
}


df["participant_id_original"] = df["participant_id"]

df["participant_id"] = df["participant_id"].replace(
    ID_CORRECTIONS
)


print("PARTICIPANT CLEANING")
print("--------------------------------")

print("Before correction:")
print(
    df["participant_id_original"].nunique()
)

print("After correction:")
print(
    df["participant_id"].nunique()
)


print("\nCorrections applied:")
for old_id, new_id in ID_CORRECTIONS.items():

    count = (
        df["participant_id_original"]
        .eq(old_id)
        .sum()
    )

    print(
        old_id,
        "->",
        new_id,
        "| recordings:",
        count
    )


print("\nUnique participants by class:")

print(
    df.groupby("class")["participant_id"]
    .nunique()
)


# Check class leakage
participant_classes = (
    df.groupby("participant_id")["class"]
    .nunique()
)

bad_ids = participant_classes[
    participant_classes > 1
]

print("\nParticipants appearing in more than one class:")
print(len(bad_ids))

if len(bad_ids) > 0:
    print(bad_ids)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved cleaned index to:")
print(OUTPUT_FILE)

