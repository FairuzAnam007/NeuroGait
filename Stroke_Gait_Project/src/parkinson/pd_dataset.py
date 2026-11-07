from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "parkinson"
    / "processed"
)


class ParkinsonDataset(Dataset):

    def __init__(self, split):

        self.X = np.load(
            DATA_DIR / f"pd_{split}_X.npy"
        ).astype(np.float32)

        self.y = np.load(
            DATA_DIR / f"pd_{split}_y.npy"
        ).astype(np.int64)

        if len(self.X) != len(self.y):
            raise ValueError("X and y lengths do not match")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = torch.from_numpy(self.X[index])
        y = torch.tensor(self.y[index], dtype=torch.long)

        return x, y


def get_pd_dataloaders(batch_size=32):

    train_dataset = ParkinsonDataset("train")
    val_dataset = ParkinsonDataset("val")
    test_dataset = ParkinsonDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        get_pd_dataloaders(batch_size=32)
    )

    X_batch, y_batch = next(iter(train_loader))

    print("PD DATALOADER TEST")
    print("-" * 40)

    print("Batch X shape:", X_batch.shape)
    print("Batch y shape:", y_batch.shape)

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print("Labels in batch:", torch.unique(y_batch))
