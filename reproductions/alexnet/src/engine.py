"""Single-epoch training and evaluation logic."""

import torch

from src.utils import AverageMeter, topk_accuracy


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch, logger=print):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    accum_steps = cfg["train"]["accum_steps"]
    amp = cfg["train"]["amp"] and device.type == "cuda"
    print_freq = cfg["train"]["print_freq"]

    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            loss = criterion(out, y)

        # Gradient accumulation: scale loss before backward
        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == total_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc1, _ = topk_accuracy(out.float(), y, topk=(1, 5))

        n = x.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(acc1, n)

        if print_freq and ((step + 1) % print_freq == 0 or (step + 1) == total_steps):
            logger(
                f"[Epoch {epoch}] step {step + 1}/{total_steps} "
                f"loss {loss_meter.avg:.4f} top1 {acc_meter.avg * 100:.2f}%"
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False):
    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    amp = amp and device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            loss = criterion(out, y)

        acc1, acc5 = topk_accuracy(out.float(), y, topk=(1, 5))
        n = x.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(acc1, n)
        top5_meter.update(acc5, n)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg