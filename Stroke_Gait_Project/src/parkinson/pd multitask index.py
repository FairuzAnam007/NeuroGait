
from pathlib import Path, PurePosixPath
from collections import defaultdict
import re
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "parkinson" / "raw"
SPLIT_DIR = ROOT / "data" / "parkinson" / "splits"
OUT_DIR = SPLIT_DIR / "multitask"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "1_10mSlow",
    "2_10mFast",
    "3_DGChangeSpeed",
    "4_DGHeadTurns",
    "5_DGPivotTurn",
    "6_DGObstacle",
    "7_DGTUG",
    "8_DGTUGTask"
]


def normalise_path(path):
    return str(path).replace("\\", "/")


def recording_key(filename):

    name = PurePosixPath(
        normalise_path(filename)
    ).stem

    subject = re.split(
        r"\s+-\s+",
        name,
        maxsplit=1
    )[0]

    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        subject
    ).upper()


def get_task(filename):
    parts = PurePosixPath(
        normalise_path(filename)
    ).parts

    for task in TASKS:
        if task.lower() in [
            part.lower() for part in parts
        ]:
            return task

    return None

splits = []

for split in ["train", "val", "test"]:

    df = pd.read_csv(
        SPLIT_DIR / f"pd_{split}.csv",
        dtype={"participant_id": str}
    )

    df["split"] = split
    splits.append(df)

participants = pd.concat(
    splits,
    ignore_index=True
)

assert len(participants) == 89
assert participants["participant_id"].is_unique
assert participants["label"].value_counts().to_dict() == {
    0: 45,
    1: 44
}

print("PARTICIPANTS:", len(participants))
print("Scanning original ZIP files...")

inventory = defaultdict(list)

for zip_path in sorted(RAW_DIR.glob("*.zip")):

    with zipfile.ZipFile(zip_path) as archive:

        for filename in archive.namelist():

            if not filename.lower().endswith(".csv"):
                continue

            task = get_task(filename)

            if task is None:
                continue

            key = (
                zip_path.resolve().as_posix().lower(),
                recording_key(filename),
                task
            )

            inventory[key].append(filename)


matched = []
missing = []
ambiguous = []

for _, row in participants.iterrows():

    zip_path = Path(str(row["zip_path"]))

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    subject_code = recording_key(
        row["csv_path"]
    )

    zip_key = zip_path.resolve().as_posix().lower()

    for task in TASKS:

        key = (
            zip_key,
            subject_code,
            task
        )

        candidates = inventory.get(key, [])

        if len(candidates) == 0:

            missing.append({
                "participant_id": row["participant_id"],
                "split": row["split"],
                "task": task
            })

            continue

        if len(candidates) > 1:

            ambiguous.append({
                "participant_id": row["participant_id"],
                "split": row["split"],
                "task": task,
                "count": len(candidates)
            })

            continue

        matched.append({
            "participant_id": row["participant_id"],
            "group": row["group"],
            "label": int(row["label"]),
            "split": row["split"],
            "task": task,
            "zip_path": str(zip_path),
            "csv_path": candidates[0]
        })


index = pd.DataFrame(matched)

slow = index[
    index["task"] == "1_10mSlow"
]

if slow["participant_id"].nunique() != 89:
    raise ValueError(
        "Slow-walking baseline matching failed. "
        "Check participant codes before continuing."
    )

assert (
    index.groupby("participant_id")["label"]
    .nunique() == 1
).all()

assert (
    index.groupby("participant_id")["split"]
    .nunique() == 1
).all()

index.to_csv(
    OUT_DIR / "pd_multitask_index.csv",
    index=False
)

pd.DataFrame(missing).to_csv(
    OUT_DIR / "pd_multitask_missing.csv",
    index=False
)

pd.DataFrame(ambiguous).to_csv(
    OUT_DIR / "pd_multitask_ambiguous.csv",
    index=False
)

print("\nMULTI-TASK MATCHING RESULTS")
print("=" * 55)

print("Unique participants:",
      index["participant_id"].nunique())

print("Matched recordings:", len(index))
print("Missing task matches:", len(missing))
print("Ambiguous task matches:", len(ambiguous))

print("\nRECORDINGS BY TASK")
print(
    pd.crosstab(
        index["task"],
        index["label"]
    ).reindex(TASKS, fill_value=0)
)

print("\nRECORDINGS BY SPLIT")
print(index["split"].value_counts())

print("\nPARTICIPANT LEAKAGE CHECK")

split_sets = {
    split: set(
        index.loc[
            index["split"] == split,
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

print("PASSED")

print("\nOutput:")
print(OUT_DIR / "pd_multitask_index.csv")
