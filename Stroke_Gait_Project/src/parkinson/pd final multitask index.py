from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MULTI_DIR = (
    ROOT / "data" / "parkinson"
    / "splits" / "multitask"
)

index = pd.read_csv(
    MULTI_DIR / "pd_multitask_index.csv",
    dtype={"participant_id": str}
)

repeats = pd.read_csv(
    MULTI_DIR / "pd_repeated_recordings_audit.csv",
    dtype={"participant_id": str}
)

print("FINAL MULTI-TASK RECORDING INDEX")
print("=" * 60)

print("Previously matched:", len(index))
print("Repeated candidates:", len(repeats))

if not repeats["valid_preliminary"].fillna(False).astype(bool).all():
    raise ValueError(
        "Some repeated recordings failed preliminary checks."
    )

participant_info = index[
    ["participant_id", "group", "label", "split"]
].drop_duplicates()

if participant_info["participant_id"].duplicated().any():
    raise ValueError(
        "Participant labels or split assignments conflict."
    )

repeats = repeats.merge(
    participant_info,
    on="participant_id",
    how="left",
    validate="many_to_one"
)

if repeats[["group", "label", "split"]].isna().any().any():
    raise ValueError(
        "Some repeated recordings have no participant match."
    )

required = [
    "participant_id",
    "group",
    "label",
    "split",
    "task",
    "zip_path",
    "csv_path"
]

final = pd.concat(
    [
        index[required],
        repeats[required]
    ],
    ignore_index=True
)

if final.duplicated(
    subset=["zip_path", "csv_path"]
).any():
    raise ValueError(
        "Duplicate recording paths found."
    )

if (
    final.groupby("participant_id")["label"]
    .nunique() > 1
).any():
    raise ValueError("Conflicting participant labels.")

if (
    final.groupby("participant_id")["split"]
    .nunique() > 1
).any():
    raise ValueError("Participant appears in multiple splits.")

split_sets = {
    split: set(
        final.loc[
            final["split"] == split,
            "participant_id"
        ]
    )
    for split in ["train", "val", "test"]
}

for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:
    assert split_sets[a].isdisjoint(split_sets[b])

assert final["participant_id"].nunique() == 89
assert len(final) == 717, (
    f"Expected 717 recordings, found {len(final)}"
)

output = MULTI_DIR / "pd_multitask_final_index.csv"

final.to_csv(output, index=False)

print("\nTOTAL RECORDINGS:", len(final))
print("TOTAL PARTICIPANTS:",
      final["participant_id"].nunique())

print("\nRECORDINGS BY TASK:")
print(
    pd.crosstab(
        final["task"],
        final["label"]
    ).to_string()
)

print("\nRECORDINGS BY SPLIT:")
print(
    final["split"].value_counts().to_string()
)

print("\nPARTICIPANTS BY SPLIT:")
print(
    final.groupby("split")["participant_id"]
    .nunique().to_string()
)

print("\nLEAKAGE CHECK: PASSED")

print("\nSaved:")
print(output)

