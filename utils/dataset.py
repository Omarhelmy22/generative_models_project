import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataloader(dataset_name="cifar10", batch_size=128, image_size=32,
                   data_dir="./data", train=True, num_workers=None):
    """Create a dataloader for the specified dataset, normalized to [-1, 1]."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def infinite_dataloader(dataloader):
    """Yields batches indefinitely by cycling through the dataloader."""
    while True:
        for batch in dataloader:
            yield batch
