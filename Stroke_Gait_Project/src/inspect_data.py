from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

print("Looking for first CSV file...")

first_csv = None

for file in RAW_DIR.rglob("*.csv"):
    first_csv = file
    break

if first_csv is None:
    print("No CSV file found.")
else:
    print("Found:")
    print(first_csv)

    df = pd.read_csv(first_csv)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

