from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLING_RATE = 36

CHANNELS = [
    "acx", "acy", "acz",
    "gyrx", "gyry", "gyrz"
]


def load_pd_recording(zip_path, csv_path):

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_path) as f:
            df = pd.read_csv(f)

    return df


def prepare_foot(df, foot_name):

    foot_labels = (
        df["foot"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sensor = df.loc[
        foot_labels == foot_name,
        ["timestamp"] + CHANNELS
    ].copy()

    for column in ["timestamp"] + CHANNELS:

        sensor[column] = pd.to_numeric(
            sensor[column],
            errors="coerce"
        )

    sensor = sensor.replace(
        [np.inf, -np.inf],
        np.nan
    )

    sensor = sensor.dropna(
        subset=["timestamp"]
    )

    sensor = sensor.sort_values(
        "timestamp"
    )

    sensor = sensor.drop_duplicates(
        subset=["timestamp"]
    )

    if sensor.empty:
        raise ValueError(
            f"No valid {foot_name} foot data"
        )

    sensor[CHANNELS] = (
        sensor[CHANNELS]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    if sensor[CHANNELS].isna().any().any():
        raise ValueError(
            f"Missing IMU values in {foot_name} foot"
        )

    return sensor


def synchronize_pd_feet(df):

    left = prepare_foot(
        df,
        "left"
    )

    right = prepare_foot(
        df,
        "right"
    )

    start = max(
        left["timestamp"].min(),
        right["timestamp"].min()
    )

    end = min(
        left["timestamp"].max(),
        right["timestamp"].max()
    )

    duration = end - start

    if duration < 5:
        raise ValueError(
            "Less than 5 seconds of common data"
        )

    number_samples = (
        int(np.floor(duration * SAMPLING_RATE)) + 1
    )

    common_time = (
        start
        + np.arange(number_samples)
        / SAMPLING_RATE
    )

    combined_channels = []

    for sensor in [left, right]:

        timestamps = (
            sensor["timestamp"]
            .to_numpy(dtype=np.float64)
        )

        for channel in CHANNELS:

            values = (
                sensor[channel]
                .to_numpy(dtype=np.float64)
            )

            interpolated = np.interp(
                common_time,
                timestamps,
                values
            )

            combined_channels.append(
                interpolated
            )

    combined = np.stack(
        combined_channels,
        axis=1
    ).astype(np.float32)

    if not np.isfinite(combined).all():
        raise ValueError(
            "Non-finite values after synchronisation"
        )

    return combined