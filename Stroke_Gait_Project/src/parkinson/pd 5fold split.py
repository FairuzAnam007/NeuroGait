from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "parkinson"
    / "processed" / "multitask"
)

CV_DIR = (
    ROOT / "data" / "parkinson"
    / "splits" / "cv_5fold"
)

CV_DIR.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(
    DATA_DIR / "pd_multitask_metadata.csv",
    dtype={"participant_id": str}
)

X = np.load(
    DATA_DIR / "pd_multitask_X_raw.npy",
    mmap_mode="r"
)

assert len(meta) == len(X)
assert X.shape == (5072, 180, 12)

label_check = meta.groupby(
    "participant_id"
)["label"].nunique()

assert (label_check == 1).all(), (
    "Conflicting participant labels!"
)

participants = (
    meta[["participant_id", "label"]]
    .drop_duplicates()
    .sort_values("participant_id")
    .reset_index(drop=True)
)

assert len(participants) == 89

print("TOTAL PARTICIPANTS:", len(participants))

print("\nPARTICIPANT CLASS DISTRIBUTION")
print(
    participants["label"]
    .value_counts()
    .sort_index()
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

participants["fold"] = 0

for fold_number, (_, test_idx) in enumerate(
    skf.split(
        participants["participant_id"],
        participants["label"]
    ),
    start=1
):
    participants.loc[
        test_idx, "fold"
    ] = fold_number

assert participants["participant_id"].is_unique
assert participants["fold"].between(1, 5).all()

print("\nFOLD DISTRIBUTION")
table = pd.crosstab(
    participants["fold"],
    participants["label"]
)

table.columns = ["Healthy", "Parkinson"]
table["Total"] = table.sum(axis=1)

print(table)

window_meta = meta.merge(
    participants[["participant_id", "fold"]],
    on="participant_id",
    how="left",
    validate="many_to_one"
)

assert len(window_meta) == len(meta)
assert window_meta["fold"].notna().all()

print("\nWINDOWS PER FOLD")

for fold in range(1, 6):

    test_mask = window_meta["fold"] == fold

    train_ids = set(
        participants.loc[
            participants["fold"] != fold,
            "participant_id"
        ]
    )

    test_ids = set(
        participants.loc[
            participants["fold"] == fold,
            "participant_id"
        ]
    )

    assert train_ids.isdisjoint(test_ids)

    print(
        f"Fold {fold}: "
        f"{len(test_ids)} test participants, "
        f"{int(test_mask.sum())} test windows"
    )

participants.to_csv(
    CV_DIR / "pd_5fold_participants.csv",
    index=False
)

window_meta.to_csv(
    CV_DIR / "pd_5fold_window_metadata.csv",
    index=False
)

print("\nPARTICIPANT LEAKAGE CHECK: PASSED")

print("\nFiles saved:")
print(CV_DIR / "pd_5fold_participants.csv")
print(CV_DIR / "pd_5fold_window_metadata.csv")

