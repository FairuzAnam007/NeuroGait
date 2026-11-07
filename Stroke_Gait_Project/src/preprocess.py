from pathlib import Path
import pandas as pd
import numpy as np


SENSOR_COLUMNS = [
    "ax", "ay", "az",
    "gx", "gy", "gz"
]

FINAL_CHANNELS = [
    "LF_ax", "LF_ay", "LF_az",
    "LF_gx", "LF_gy", "LF_gz",

    "RF_ax", "RF_ay", "RF_az",
    "RF_gx", "RF_gy", "RF_gz"
]


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


def interpolate_sensor(
    df,
    common_time,
    prefix
):

    result = {}

    original_time = df["T"].to_numpy()

    for column in SENSOR_COLUMNS:

        result[
            f"{prefix}_{column}"
        ] = np.interp(
            common_time,
            original_time,
            df[column].to_numpy()
        )

    return result


def synchronize_feet(
    left_path,
    right_path
):

    left = load_sensor(left_path)
    right = load_sensor(right_path)


    start_time = max(
        left["T"].min(),
        right["T"].min()
    )

    end_time = min(
        left["T"].max(),
        right["T"].max()
    )

    if start_time >= end_time:

        raise ValueError(
            "No overlapping time range."
        )


    left_dt = np.median(
        np.diff(left["T"].to_numpy())
    )

    right_dt = np.median(
        np.diff(right["T"].to_numpy())
    )

    time_step = np.median(
        [left_dt, right_dt]
    )


    common_time = np.arange(
        start_time,
        end_time,
        time_step
    )

    left_sync = interpolate_sensor(
        left,
        common_time,
        "LF"
    )

    right_sync = interpolate_sensor(
        right,
        common_time,
        "RF"
    )

    combined = pd.DataFrame(
        {
            **left_sync,
            **right_sync
        }
    )

    combined = combined[
        FINAL_CHANNELS
    ]

    return combined