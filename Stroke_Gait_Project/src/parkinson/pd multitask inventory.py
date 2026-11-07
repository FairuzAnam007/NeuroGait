from pathlib import Path
from collections import Counter
import zipfile

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "parkinson" / "raw"

TASKS = [
    "1_10mSlow",
    "2_10mFast",
    "3_DGChangeSpeed",
    "4_DGHeadTurns",
    "5_DGPivotTurn",
    "6_DGObstacle",
    "7_DGTUG",
    "8_DGTUGTask"
]

zip_files = sorted(RAW_DIR.glob("*.zip"))

print("PD MULTI-TASK ZIP INVENTORY")
print("=" * 55)
print("ZIP files found:", len(zip_files))

counts = {
    "Healthy": Counter(),
    "Parkinson": Counter()
}

examples = {
    "Healthy": {},
    "Parkinson": {}
}

for zip_path in zip_files:

    zip_name = zip_path.name.lower()

    if "healthy" in zip_name:
        group = "Healthy"
    elif "parkinson" in zip_name or "patients" in zip_name:
        group = "Parkinson"
    else:
        print("Skipping unknown ZIP:", zip_path.name)
        continue

    print("\nScanning:", zip_path.name)

    with zipfile.ZipFile(zip_path) as archive:

        for filename in archive.namelist():

            if not filename.lower().endswith(".csv"):
                continue

            path_text = filename.replace("\\", "/").lower()

            for task in TASKS:

                if task.lower() in path_text:

                    counts[group][task] += 1

                    if task not in examples[group]:
                        examples[group][task] = filename

                    break


print("\n" + "=" * 55)
print("TASK COVERAGE")
print("=" * 55)

print(
    f"{'Task':<22} {'Healthy':>10} {'Parkinson':>12}"
)

for task in TASKS:

    print(
        f"{task:<22} "
        f"{counts['Healthy'][task]:>10} "
        f"{counts['Parkinson'][task]:>12}"
    )


print("\n" + "=" * 55)
print("EXAMPLE FILE PATHS")
print("=" * 55)

for group in ["Healthy", "Parkinson"]:

    print("\nGROUP:", group)

    for task in TASKS:

        print("\nTASK:", task)

        print(
            examples[group].get(
                task,
                "NOT FOUND"
            )
        )
