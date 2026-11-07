from pathlib import Path
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

csv_files = list(RAW_DIR.rglob("*.csv"))

full_ids = set()
base_ids = set()

base_to_full = defaultdict(set)


for file in csv_files:


    first_part = file.stem.split("_")[0]

    full_ids.add(first_part)

    match = re.match(r"([A-Za-z]+\d+)", first_part)

    if match:
        base_id = match.group(1)
    else:
        base_id = first_part

    base_ids.add(base_id)

    base_to_full[base_id].add(first_part)


print("PARTICIPANT ID CHECK")
print("--------------------------------")

print("Full ID count:", len(full_ids))
print("Base ID count:", len(base_ids))


print("\nBASE IDs WITH MULTIPLE FULL IDs")
print("--------------------------------")

duplicate_count = 0

for base_id, variants in sorted(base_to_full.items()):

    if len(variants) > 1:

        duplicate_count += 1

        print(
            base_id,
            "->",
            sorted(variants)
        )


print("\nNumber of base IDs with multiple variants:",
      duplicate_count)


print("\nFIRST 20 FULL IDs")
print("--------------------------------")

for pid in sorted(full_ids)[:20]:
    print(pid)


print("\nFIRST 20 BASE IDs")
print("--------------------------------")

for pid in sorted(base_ids)[:20]:
    print(pid)


