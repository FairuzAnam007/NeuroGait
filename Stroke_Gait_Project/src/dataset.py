from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class GaitDataset(Dataset):

    def __init__(self, X_path, y_path):

        self.X = np.load(X_path)
        self.y = np.load(y_path)

        self.X = torch.tensor(
            self.X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            self.y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_dataloaders(batch_size=32):

    train_dataset = GaitDataset(
        DATA_DIR / "train_X.npy",
        DATA_DIR / "train_y.npy"
    )

    val_dataset = GaitDataset(
        DATA_DIR / "val_X.npy",
        DATA_DIR / "val_y.npy"
    )

    test_dataset = GaitDataset(
        DATA_DIR / "test_X.npy",
        DATA_DIR / "test_y.npy"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32
    )

    X_batch, y_batch = next(
        iter(train_loader)
    )

    print("DATALOADER TEST")
    print("------------------------------")

    print("Batch X shape:")
    print(X_batch.shape)

    print("\nBatch y shape:")
    print(y_batch.shape)

    print("\nFirst labels:")
    print(y_batch[:10])

    print("\nTrain batches:")
    print(len(train_loader))

    print("\nValidation batches:")
    print(len(val_loader))

    print("\nTest batches:")
    print(len(test_loader))

