from pathlib import Path

import pandas as pd
import numpy as np

from pd_preprocess import (
    load_pd_recording,
    synchronize_pd_feet
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "parkinson"
    / "splits"
)

index = pd.read_csv(
    SPLIT_DIR / "pd_train.csv"
)

for group in ["Healthy", "Parkinson"]:

    row = index[
        index["group"] == group
    ].iloc[0]

    df = load_pd_recording(
        row["zip_path"],
        row["csv_path"]
    )

    combined = synchronize_pd_feet(df)

    print("\n" + "=" * 45)
    print("GROUP:", group)
    print("PARTICIPANT:", row["participant_id"])
    print("Original shape:", df.shape)
    print("Synchronized shape:", combined.shape)
    print("Channels:", combined.shape[1])
    print("Missing values:", np.isnan(combined).sum())
    print("All finite:", np.isfinite(combined).all())
