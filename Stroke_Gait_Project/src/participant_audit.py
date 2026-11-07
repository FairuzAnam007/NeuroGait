from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "processed" / "recording_index.csv"

df = pd.read_csv(INDEX_FILE)

print("PARTICIPANT AUDIT")
print("--------------------------------")

healthy_ids = sorted(
    df.loc[df["class"] == "Healthy", "participant_id"].unique()
)

stroke_ids = sorted(
    df.loc[df["class"] == "Stroke", "participant_id"].unique()
)

print("Healthy unique IDs:", len(healthy_ids))
print("Stroke unique IDs:", len(stroke_ids))

print("\nTotal unique IDs:")
print(df["participant_id"].nunique())

overlap_ids = sorted(
    set(healthy_ids).intersection(stroke_ids)
)

print("\nIDs appearing in BOTH Healthy and Stroke:")
print(len(overlap_ids))

for pid in overlap_ids:
    print(pid)

print("\n--------------------------------")
print("BASE IDs WITH MULTIPLE RAW IDs")
print("--------------------------------")

variant_table = (
    df.groupby("participant_id")["raw_id"]
    .apply(lambda x: sorted(set(x)))
)

variant_count = 0

for participant_id, raw_ids in variant_table.items():

    if len(raw_ids) > 1:

        variant_count += 1

        print(
            participant_id,
            "->",
            raw_ids
        )

print("\nNumber of participant IDs with multiple raw variants:")
print(variant_count)

print("\n--------------------------------")
print("PARTICIPANTS WITH MOST RECORDINGS")
print("--------------------------------")

record_counts = (
    df.groupby(["participant_id", "class"])
    .size()
    .reset_index(name="recordings")
    .sort_values("recordings", ascending=False)
)

print(record_counts.head(30).to_string(index=False))


print("\n--------------------------------")
print("HEALTHY IDS")
print("--------------------------------")
print(healthy_ids)

