"""CIFAR-10 data loading and augmentation."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 training set statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = data_cfg.get("pin_memory", True)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Augmentation: random crop
        transforms.RandomHorizontalFlip(),          # Augmentation: horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,              # Avoid incomplete batch with gradient accumulation
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader