"""Training entry point.

Usage:
  python -m src.train --config configs/default.yaml
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.dataset import build_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import AlexNetCIFAR
from src.utils import load_config, save_checkpoint, save_history, set_seed


def build_scheduler(optimizer, cfg, epochs):
    scfg = cfg["scheduler"]
    name = scfg.get("name", "cosine")
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return StepLR(optimizer, step_size=scfg["step_size"], gamma=scfg["gamma"])
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"]["amp"] and device.type == "cuda"
    print(f"Device: {device} | AMP: {amp}")

    # Data
    train_loader, test_loader = build_dataloaders(cfg)

    # Model / loss / optimizer
    model = AlexNetCIFAR(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, cfg["train"]["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    history = []
    best_top1 = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, test_loader, criterion, device, amp=amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{cfg['train']['epochs']} | "
            f"train_loss {train_loss:.4f} train_top1 {train_top1*100:.2f}% | "
            f"val_loss {val_loss:.4f} val_top1 {val_top1*100:.2f}% val_top5 {val_top5*100:.2f}% | "
            f"lr {lr_now:.5f} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": lr_now,
            "time_s": elapsed,
        })
        save_history(history, os.path.join(output_dir, "history.json"))

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_top1": best_top1,
            "config": cfg,
        }

        # Always save last checkpoint for resuming
        save_checkpoint(state, os.path.join(output_dir, "last.pth"))

        # Save best checkpoint
        if val_top1 > best_top1:
            best_top1 = val_top1
            state["best_top1"] = best_top1
            save_checkpoint(state, os.path.join(output_dir, "best.pth"))
            print(f"  -> New best val_top1: {best_top1*100:.2f}%, saved best.pth")

    print(f"Training finished. Best val_top1 = {best_top1*100:.2f}%")


if __name__ == "__main__":
    main()