from pathlib import Path
import zipfile
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "parkinson" / "raw"

zip_files = sorted(RAW_DIR.glob("*.zip"))

print("TOTAL ZIP FILES:", len(zip_files))

for zip_path in zip_files:

    print("\n" + "=" * 60)
    print("ZIP:", zip_path.name)

    with zipfile.ZipFile(zip_path, "r") as z:

        files = [
            name for name in z.namelist()
            if not name.endswith("/")
        ]

        csv_files = [
            name for name in files
            if name.lower().endswith(".csv")
        ]

        print("Total files:", len(files))
        print("CSV files:", len(csv_files))

        print("\nFirst 5 CSV files:")

        for name in csv_files[:5]:
            print(" ", name)
