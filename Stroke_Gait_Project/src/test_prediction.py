from pathlib import Path
import pandas as pd

from predict import predict_gait


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recording_index_clean.csv"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

df = pd.read_csv(INDEX_FILE)

first_row = df.iloc[0]


def get_local_path(old_path):

    old_path = str(old_path).replace("\\", "/")

    marker = "data/raw/"

    if marker not in old_path:
        raise ValueError(
            f"Cannot locate data/raw in path: {old_path}"
        )

    relative_path = old_path.split(marker, 1)[1]

    return RAW_DIR / Path(relative_path)


left_file = get_local_path(
    first_row["left_foot_path"]
)

right_file = get_local_path(
    first_row["right_foot_path"]
)


print("TEST RECORDING")
print("--------------------------------")

print("Participant:", first_row["participant_id"])
print("Actual Class:", first_row["class"])
print("Recording:", first_row["recording_id"])

print("\nUsing files:")
print("Left :", left_file)
print("Right:", right_file)


if not left_file.is_file():
    raise FileNotFoundError(
        f"Left Foot CSV missing:\n{left_file}"
    )

if not right_file.is_file():
    raise FileNotFoundError(
        f"Right Foot CSV missing:\n{right_file}"
    )


result = predict_gait(
    left_file,
    right_file
)


print("\nFINAL PREDICTION")
print("--------------------------------")

print("Prediction:", result["prediction"])

print(
    f'Healthy Probability: '
    f'{result["healthy_probability"]:.2f}%'
)

print(
    f'Stroke Probability: '
    f'{result["stroke_probability"]:.2f}%'
)

print(
    f'Confidence: '
    f'{result["confidence"]:.2f}%'
)

print(
    "Number of windows:",
    result["windows"]
)