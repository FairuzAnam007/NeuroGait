
from pathlib import Path, PurePosixPath
import re
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "parkinson" / "splits"
MULTI_DIR = SPLIT_DIR / "multitask"

ambiguous = pd.read_csv(
    MULTI_DIR / "pd_multitask_ambiguous.csv",
    dtype={"participant_id": str}
)

if ambiguous.empty:
    print("No ambiguous recordings found.")
    raise SystemExit

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


print("AMBIGUOUS RECORDING INSPECTION")
print("=" * 60)

for number, row in ambiguous.iterrows():

    pid = str(row["participant_id"])
    task = str(row["task"])

    person = splits[
        splits["participant_id"] == pid
    ].iloc[0]

    zip_path = Path(person["zip_path"])

    subject_code = recording_key(
        person["csv_path"]
    )

    print(f"\nCASE {number + 1}")
    print("Participant:", pid)
    print("Group:", person["group"])
    print("Task:", task)
    print("ZIP:", zip_path.name)
    print("Subject code:", subject_code)

    print("\nMatching CSV files:")

    candidates = []

    with zipfile.ZipFile(zip_path) as archive:

        for filename in archive.namelist():

            if not filename.lower().endswith(".csv"):
                continue

            parts = PurePosixPath(
                filename.replace("\\", "/")
            ).parts

            correct_task = any(
                part.lower() == task.lower()
                for part in parts
            )

            if (
                correct_task
                and recording_key(filename) == subject_code
            ):
                candidates.append(filename)

    for i, filename in enumerate(candidates, 1):
        print(f"  {i}. {filename}")

    print("Candidate count:", len(candidates))
