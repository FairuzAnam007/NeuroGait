from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("NORMALIZATION")
print("--------------------------------")


X_train = np.load(
    DATA_DIR / "train_X_raw.npy"
)

X_val = np.load(
    DATA_DIR / "val_X_raw.npy"
)

X_test = np.load(
    DATA_DIR / "test_X_raw.npy"
)


print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

std[std == 0] = 1.0


print("\nMean shape:")
print(mean.shape)

print("Std shape:")
print(std.shape)


X_train_norm = (
    X_train - mean
) / std

X_val_norm = (
    X_val - mean
) / std

X_test_norm = (
    X_test - mean
) / std

np.save(
    DATA_DIR / "train_X.npy",
    X_train_norm.astype(np.float32)
)

np.save(
    DATA_DIR / "val_X.npy",
    X_val_norm.astype(np.float32)
)

np.save(
    DATA_DIR / "test_X.npy",
    X_test_norm.astype(np.float32)
)

np.save(
    MODEL_DIR / "channel_mean.npy",
    mean
)

np.save(
    MODEL_DIR / "channel_std.npy",
    std
)


print("\nNormalized training mean:")
print(
    X_train_norm.mean(
        axis=(0, 1)
    )
)

print("\nNormalized training std:")
print(
    X_train_norm.std(
        axis=(0, 1)
    )
)


print("\nSaved:")
print("train_X.npy")
print("val_X.npy")
print("test_X.npy")
print("channel_mean.npy")
print("channel_std.npy")
