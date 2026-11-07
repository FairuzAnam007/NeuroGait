from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "parkinson" / "splits"
MULTI_DIR = SPLIT_DIR / "multitask"

ambiguous = pd.read_csv(
    MULTI_DIR / "pd_multitask_ambiguous.csv",
    dtype={"participant_id": str}
)

splits = pd.concat(
    [
        pd.read_csv(
            SPLIT_DIR / f"pd_{split}.csv",
            dtype={"participant_id": str}
        )
        for split in ["train", "val", "test"]
    ],
    ignore_index=True
)

CHANNELS = [
    "acx", "acy", "acz",
    "gyrx", "gyry", "gyrz"
]


def recording_key(filename):
    name = PurePosixPath(
        str(filename).replace("\\", "/")
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


records = []

for _, row in ambiguous.iterrows():

    pid = str(row["participant_id"])
    task = str(row["task"])

    person = splits[
        splits["participant_id"] == pid
    ].iloc[0]

    zip_path = Path(person["zip_path"])
    subject_code = recording_key(
        person["csv_path"]
    )

    with zipfile.ZipFile(zip_path) as archive:

        for filename in archive.namelist():

            if not filename.lower().endswith(".csv"):
                continue

            parts = PurePosixPath(
                filename.replace("\\", "/")
            ).parts

            if task.lower() not in [
                p.lower() for p in parts
            ]:
                continue

            if recording_key(filename) != subject_code:
                continue

            try:
                with archive.open(filename) as file:
                    df = pd.read_csv(file)

                required = ["foot", "timestamp"] + CHANNELS

                if not set(required).issubset(df.columns):
                    raise ValueError(
                        "Missing required columns"
                    )

                foot_counts = df["foot"].value_counts()
                time_values = pd.to_numeric(
                    df["timestamp"], errors="coerce"
                )

                valid_times = time_values.dropna()

                duration = (
                    float(valid_times.max() - valid_times.min())
                    if len(valid_times) > 1
                    else 0.0
                )

                valid = (
                    len(df) > 0
                    and len(valid_times) > 1
                    and duration > 5
                    and df[CHANNELS].notna().all().all()
                    and len(foot_counts) >= 2
                )

                records.append({
                    "participant_id": pid,
                    "task": task,
                    "zip_path": str(zip_path),
                    "csv_path": filename,
                    "rows": len(df),
                    "duration_raw": round(duration, 3),
                    "foot_values": str(
                        foot_counts.to_dict()
                    ),
                    "valid_preliminary": bool(valid),
                    "error": ""
                })

            except Exception as error:

                records.append({
                    "participant_id": pid,
                    "task": task,
                    "zip_path": str(zip_path),
                    "csv_path": filename,
                    "rows": 0,
                    "duration_raw": 0,
                    "foot_values": "",
                    "valid_preliminary": False,
                    "error": str(error)
                })


result = pd.DataFrame(records)

print("REPEATED RECORDING VALIDATION")
print("=" * 65)

for _, row in result.iterrows():

    print("\nParticipant:", row["participant_id"])
    print("Task:", row["task"])
    print("File:", Path(row["csv_path"]).name)
    print("Rows:", row["rows"])
    print("Raw timestamp range:", row["duration_raw"])
    print("Foot values:", row["foot_values"])
    print("Preliminary valid:", row["valid_preliminary"])

    if row["error"]:
        print("Error:", row["error"])


output = MULTI_DIR / "pd_repeated_recordings_audit.csv"
result.to_csv(output, index=False)

print("\nTotal candidates:", len(result))
print("Preliminary valid:", result["valid_preliminary"].sum())
print("Saved:", output)
