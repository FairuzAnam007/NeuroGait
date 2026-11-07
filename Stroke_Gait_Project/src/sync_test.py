from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train_recordings.csv"
)

SAMPLING_RATE = 104

SENSOR_COLUMNS = [
    "ax", "ay", "az",
    "gx", "gy", "gz"
]


index_df = pd.read_csv(TRAIN_FILE)

row = index_df.iloc[0]

left_path = Path(row["left_foot_path"])
right_path = Path(row["right_foot_path"])

print("TIMESTAMP SYNCHRONISATION TEST")
print("--------------------------------")

print("Participant:", row["participant_id"])
print("Class:", row["class"])
print("Recording:", row["recording_id"])


def load_sensor(path):

    df = pd.read_csv(path)

    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    df = df[
        ["T"] + SENSOR_COLUMNS
    ].copy()


    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["T"]
    )


    df = df.sort_values("T")


    df = df.drop_duplicates(
        subset="T",
        keep="first"
    )


    df[SENSOR_COLUMNS] = (
        df[SENSOR_COLUMNS]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    df = df.dropna()

    return df.reset_index(drop=True)


left = load_sensor(left_path)
right = load_sensor(right_path)


print("\nOriginal Left rows:", len(left))
print("Original Right rows:", len(right))

print("\nLeft T:")
print(left["T"].min(), "to", left["T"].max())

print("Right T:")
print(right["T"].min(), "to", right["T"].max())


start_time = max(
    left["T"].min(),
    right["T"].min()
)

end_time = min(
    left["T"].max(),
    right["T"].max()
)

print("\nCommon start:", start_time)
print("Common end:", end_time)


if start_time >= end_time:
    raise ValueError(
        "Left and Right Foot have no overlapping time range."
    )


left_dt = np.median(
    np.diff(left["T"].values)
)

right_dt = np.median(
    np.diff(right["T"].values)
)

original_dt = np.median(
    [left_dt, right_dt]
)

print("\nMedian Left timestamp step:", left_dt)
print("Median Right timestamp step:", right_dt)
print("Median combined timestamp step:", original_dt)


time_step = original_dt

common_time = np.arange(
    start_time,
    end_time,
    time_step
)

print("\nCommon samples:")
print(len(common_time))


def interpolate_to_grid(
    df,
    common_time,
    prefix
):

    output = {
        "T": common_time
    }

    original_time = df["T"].values

    for column in SENSOR_COLUMNS:

        output[
            f"{prefix}_{column}"
        ] = np.interp(
            common_time,
            original_time,
            df[column].values
        )

    return pd.DataFrame(output)


left_sync = interpolate_to_grid(
    left,
    common_time,
    "LF"
)

right_sync = interpolate_to_grid(
    right,
    common_time,
    "RF"
)


combined = pd.DataFrame({
    "T": common_time
})

for column in left_sync.columns:

    if column != "T":
        combined[column] = left_sync[column]

for column in right_sync.columns:

    if column != "T":
        combined[column] = right_sync[column]


CHANNELS = [
    "LF_ax", "LF_ay", "LF_az",
    "LF_gx", "LF_gy", "LF_gz",

    "RF_ax", "RF_ay", "RF_az",
    "RF_gx", "RF_gy", "RF_gz"
]


print("\n--------------------------------")
print("SYNCHRONISED BILATERAL DATA")
print("--------------------------------")

print("Shape:")
print(combined.shape)

print("\n12 model channels:")
print(combined[CHANNELS].columns.tolist())

print("\nMissing values:")
print(
    combined[CHANNELS]
    .isnull()
    .sum()
)

print("\nFirst 5 rows:")
print(
    combined[
        ["T"] + CHANNELS
    ].head()
)

print("\nLeft rows after sync:", len(left_sync))
print("Right rows after sync:", len(right_sync))

print("\nSame length:",
      len(left_sync) == len(right_sync))

