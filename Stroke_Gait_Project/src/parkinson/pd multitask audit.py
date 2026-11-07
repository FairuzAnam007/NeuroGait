
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "parkinson" / "splits"

tasks = [
    "1_10mSlow",
    "2_10mFast",
    "3_DGChangeSpeed",
    "4_DGHeadTurns",
    "5_DGPivotTurn",
    "6_DGObstacle",
    "7_DGTUG",
    "8_DGTUGTask"
]

print("PARKINSON MULTI-TASK AUDIT")
print("=" * 55)

for split in ["train", "val", "test"]:

    df = pd.read_csv(
        SPLIT_DIR / f"pd_{split}.csv"
    )

    print(f"\nSPLIT: {split.upper()}")
    print("Participants:", df["participant_id"].nunique())
    print("Recordings:", len(df))

    print("\nExisting labels:")
    print(df["label"].value_counts().sort_index())

    print("\nTasks in current split:")
    if "task" in df.columns:
        print(df["task"].value_counts())
    else:
        print("Task column not present")

print("\nEXPECTED MULTI-TASK CATEGORIES")

for task in tasks:
    print(task)
