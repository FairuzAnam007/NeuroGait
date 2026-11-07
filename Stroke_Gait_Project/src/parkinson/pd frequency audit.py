from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "parkinson" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "parkinson" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []

for split in ["train", "val", "test"]:

    index = pd.read_csv(
        SPLIT_DIR / f"pd_{split}.csv"
    )

    for _, row in index.iterrows():

        with zipfile.ZipFile(row["zip_path"]) as z:
            with z.open(row["csv_path"]) as f:
                df = pd.read_csv(
                    f,
                    usecols=["foot", "timestamp"]
                )

        df["foot"] = (
            df["foot"].astype(str).str.strip().str.lower()
        )

        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )

        result = {
            "participant_id": row["participant_id"],
            "group": row["group"],
            "split": split,
        }

        time_ranges = []

        for foot in ["left", "right"]:

            t = (
                df.loc[df["foot"] == foot, "timestamp"]
                .dropna()
                .sort_values()
                .drop_duplicates()
                .to_numpy(dtype=float)
            )

            dt = np.diff(t)
            dt = dt[dt > 0]

            result[f"{foot}_rows"] = len(t)

            result[f"{foot}_hz"] = (
                1 / np.median(dt)
                if len(dt) else np.nan
            )

            if len(t):
                time_ranges.append((t[0], t[-1]))

        if len(time_ranges) == 2:

            start = max(x[0] for x in time_ranges)
            end = min(x[1] for x in time_ranges)

            result["overlap_seconds"] = max(0, end - start)

        else:
            result["overlap_seconds"] = 0

        results.append(result)


audit = pd.DataFrame(results)

output = OUTPUT_DIR / "pd_frequency_audit.csv"
audit.to_csv(output, index=False)

print("PD FREQUENCY AUDIT")
print("-" * 45)

print("Total recordings:", len(audit))

for foot in ["left", "right"]:

    values = audit[f"{foot}_hz"].dropna()

    print(f"\n{foot.upper()} FOOT")

    print("Minimum Hz:", round(values.min(), 2))
    print("Median Hz:", round(values.median(), 2))
    print("Maximum Hz:", round(values.max(), 2))

print("\nCOMMON DURATION")
print(
    "Minimum:",
    round(audit["overlap_seconds"].min(), 2),
    "seconds"
)

print(
    "Median:",
    round(audit["overlap_seconds"].median(), 2),
    "seconds"
)

print(
    "Recordings with at least 5 seconds:",
    (audit["overlap_seconds"] >= 5).sum()
)

print("\nSaved:", output)
