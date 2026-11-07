from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "parkinson" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "parkinson"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


train_X = np.load(DATA_DIR / "pd_train_X_raw.npy")

print("TRAINING DATA:", train_X.shape)


mean = train_X.mean(axis=(0, 1))
std = train_X.std(axis=(0, 1))

std = np.where(std < 1e-6, 1.0, std)

np.save(MODEL_DIR / "pd_channel_mean.npy", mean)
np.save(MODEL_DIR / "pd_channel_std.npy", std)

print("\nCHANNEL MEANS:")
print(mean)

print("\nCHANNEL STANDARD DEVIATIONS:")
print(std)

for split in ["train", "val", "test"]:

    X = np.load(
        DATA_DIR / f"pd_{split}_X_raw.npy"
    )

    X_normalized = (
        (X - mean) / std
    ).astype(np.float32)

    if not np.isfinite(X_normalized).all():
        raise ValueError(
            f"Invalid values found in {split}"
        )

    output = DATA_DIR / f"pd_{split}_X.npy"
    np.save(output, X_normalized)

    print(f"\n{split.upper()}")
    print("Shape:", X_normalized.shape)
    print("Mean:", round(float(X_normalized.mean()), 4))
    print("Std:", round(float(X_normalized.std()), 4))
    print("Saved:", output)
