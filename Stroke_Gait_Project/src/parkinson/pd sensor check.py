from pathlib import Path
from collections import Counter
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "parkinson" / "raw"

zip_files = sorted(RAW_DIR.glob("*.zip"))

print("TOTAL ZIP FILES:", len(zip_files))

group_counts = Counter()
total_csv = 0
seen_names = set()
duplicate_names = []

for zip_path in zip_files:

    zip_name = zip_path.name.lower()

    if "healthy" in zip_name:
        group = "Healthy"
    elif "patients" in zip_name:
        group = "Parkinson"
    else:
        group = "Other"

    with zipfile.ZipFile(zip_path, "r") as z:

        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        group_counts[group] += len(csv_files)
        total_csv += len(csv_files)

        for name in csv_files:
            filename = Path(name).name.lower()

            if filename in seen_names:
                duplicate_names.append(filename)

            seen_names.add(filename)


print("\nCSV RECORDING COUNTS")
print("--------------------------------")
print("Healthy:", group_counts["Healthy"])
print("Parkinson:", group_counts["Parkinson"])
print("Other:", group_counts["Other"])
print("Total:", total_csv)

print("\nRepeated filenames:", len(duplicate_names))


for target_group in ["Healthy", "Parkinson"]:

    for zip_path in zip_files:

        name_lower = zip_path.name.lower()

        match = (
            ("healthy" in name_lower)
            if target_group == "Healthy"
            else ("patients" in name_lower)
        )

        if not match:
            continue

        with zipfile.ZipFile(zip_path, "r") as z:

            csv_files = [
                name for name in z.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                continue

            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, nrows=10)

            print("\n" + "=" * 55)
            print("GROUP:", target_group)
            print("ZIP:", zip_path.name)
            print("SAMPLE:", csv_files[0])
            print("COLUMNS:", df.columns.tolist())
            print("SHAPE OF PREVIEW:", df.shape)

            print("\nFIRST 3 ROWS:")
            print(df.head(3).to_string(index=False))

            if "foot" in df.columns:
                print("\nFOOT VALUES:")
                print(df["foot"].unique())

            break
