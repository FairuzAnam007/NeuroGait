from pathlib import Path
import pandas as pd
import numpy as np

from preprocess import synchronize_feet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLING_RATE = 104

WINDOW_SECONDS = 5

WINDOW_SIZE = (
    SAMPLING_RATE
    * WINDOW_SECONDS
)

OVERLAP = 0.50

STEP_SIZE = int(
    WINDOW_SIZE
    * (1 - OVERLAP)
)


CLASS_TO_LABEL = {
    "Healthy": 0,
    "Stroke": 1
}


def create_windows(data):

    values = data.to_numpy(
        dtype=np.float32
    )

    windows = []

    for start in range(
        0,
        len(values) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        window = values[
            start:end
        ]

        windows.append(
            window
        )

    return windows


def process_split(split_name):

    input_file = (
        SPLIT_DIR
        / f"{split_name}_recordings.csv"
    )

    df = pd.read_csv(
        input_file
    )

    print("\n================================")
    print(split_name.upper())
    print("================================")

    print(
        "Recordings:",
        len(df)
    )

    all_windows = []
    all_labels = []
    all_participants = []
    all_recordings = []

    failed_recordings = []

    for index, row in df.iterrows():

        participant_id = str(
            row["participant_id"]
        )

        recording_id = str(
            row["recording_id"]
        )

        class_name = row["class"]

        label = CLASS_TO_LABEL[
            class_name
        ]

        try:

            combined = synchronize_feet(
                Path(row["left_foot_path"]),
                Path(row["right_foot_path"])
            )

            windows = create_windows(
                combined
            )

            if len(windows) == 0:

                print(
                    "Skipped - too short:",
                    recording_id
                )

                failed_recordings.append(
                    recording_id
                )

                continue

            all_windows.extend(
                windows
            )

            all_labels.extend(
                [label] * len(windows)
            )

            all_participants.extend(
                [participant_id] * len(windows)
            )

            all_recordings.extend(
                [recording_id] * len(windows)
            )

            print(
                f"[{index + 1}/{len(df)}]",
                participant_id,
                class_name,
                "->",
                len(windows),
                "windows"
            )

        except Exception as error:

            print(
                "FAILED:",
                recording_id
            )

            print(
                "Reason:",
                error
            )

            failed_recordings.append(
                recording_id
            )


    X = np.asarray(
        all_windows,
        dtype=np.float32
    )

    y = np.asarray(
        all_labels,
        dtype=np.int64
    )

    participants = np.asarray(
        all_participants
    )

    recordings = np.asarray(
        all_recordings
    )


    np.save(
        OUTPUT_DIR / f"{split_name}_X_raw.npy",
        X
    )

    np.save(
        OUTPUT_DIR / f"{split_name}_y.npy",
        y
    )

    np.save(
        OUTPUT_DIR / f"{split_name}_participants.npy",
        participants
    )

    np.save(
        OUTPUT_DIR / f"{split_name}_recordings.npy",
        recordings
    )


    print("\n--------------------------------")
    print(split_name.upper(), "SUMMARY")
    print("--------------------------------")

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Healthy windows:",
        np.sum(y == 0)
    )

    print(
        "Stroke windows:",
        np.sum(y == 1)
    )

    print(
        "Unique participants:",
        len(np.unique(participants))
    )

    print(
        "Failed recordings:",
        len(failed_recordings)
    )

    if failed_recordings:

        print(
            "\nFailed recording IDs:"
        )

        for item in failed_recordings:
            print(item)

process_split("train")

process_split("val")

process_split("test")


print("\n================================")
print("ALL SPLITS PROCESSED")
print("================================")