from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

csv_files = list(RAW_DIR.rglob("*.csv"))

print("FIRST 50 FILENAMES")
print("---------------------------------------------")

for i, file in enumerate(csv_files[:50], start=1):
    print(f"{i}. {file.name}")

print("\n---------------------------------------------")
print("TOTAL CSV FILES:", len(csv_files))
