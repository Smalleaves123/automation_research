"""Test-set evaluation entry point.

Usage:
  python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best.pth
"""

import argparse
import os

import torch
import torch.nn as nn

from src.dataset import build_dataloaders
from src.engine import evaluate
from src.model import AlexNetCIFAR
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pth")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"]["amp"] and device.type == "cuda"

    model = AlexNetCIFAR(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(
        f"Loaded {args.checkpoint} "
        f"(epoch {ckpt.get('epoch')}, best_top1 {ckpt.get('best_top1', 0)*100:.2f}%)"
    )

    _, test_loader = build_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()

    test_loss, top1, top5 = evaluate(model, test_loader, criterion, device, amp=amp)
    print(f"Test | loss {test_loss:.4f} | top1 {top1*100:.2f}% | top5 {top5*100:.2f}%")


if __name__ == "__main__":
    main()