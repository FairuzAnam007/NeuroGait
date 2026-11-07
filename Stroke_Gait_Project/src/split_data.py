from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recording_index_clean.csv"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT_FILE)


participants = (
    df[
        [
            "participant_id",
            "class"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("PARTICIPANT LEVEL SPLIT")
print("--------------------------------")

print("Total participants:")
print(len(participants))

print("\nClass counts:")
print(
    participants["class"].value_counts()
)


train_participants, temp_participants = train_test_split(
    participants,
    test_size=0.30,
    random_state=42,
    stratify=participants["class"]
)


val_participants, test_participants = train_test_split(
    temp_participants,
    test_size=0.50,
    random_state=42,
    stratify=temp_participants["class"]
)

train_ids = set(
    train_participants["participant_id"]
)

val_ids = set(
    val_participants["participant_id"]
)

test_ids = set(
    test_participants["participant_id"]
)

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)


train_df = df[
    df["participant_id"].isin(train_ids)
].copy()

val_df = df[
    df["participant_id"].isin(val_ids)
].copy()

test_df = df[
    df["participant_id"].isin(test_ids)
].copy()


train_df.to_csv(
    SPLIT_DIR / "train_recordings.csv",
    index=False
)

val_df.to_csv(
    SPLIT_DIR / "val_recordings.csv",
    index=False
)

test_df.to_csv(
    SPLIT_DIR / "test_recordings.csv",
    index=False
)


train_participants.to_csv(
    SPLIT_DIR / "train_participants.csv",
    index=False
)

val_participants.to_csv(
    SPLIT_DIR / "val_participants.csv",
    index=False
)

test_participants.to_csv(
    SPLIT_DIR / "test_participants.csv",
    index=False
)


def show_split(name, participant_df, recording_df):

    print(
        f"\n{name}"
    )

    print(
        "-" * 30
    )

    print(
        "Participants:",
        len(participant_df)
    )

    print(
        participant_df[
            "class"
        ].value_counts()
    )

    print(
        "Recordings:",
        len(recording_df)
    )


show_split(
    "TRAIN",
    train_participants,
    train_df
)

show_split(
    "VALIDATION",
    val_participants,
    val_df
)

show_split(
    "TEST",
    test_participants,
    test_df
)


print(
    "\nParticipant overlap check: PASSED"
)

print(
    "\nSplit files saved to:"
)

print(
    SPLIT_DIR
)
