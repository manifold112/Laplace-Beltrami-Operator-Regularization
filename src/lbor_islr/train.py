import argparse
import os
import time
import random
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lbor_islr.losses import LBORLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for LBOR-based skeleton ISLR."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory defined in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed defined in config.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} (avg:{avg" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk=(1,),
) -> Tuple[torch.Tensor, ...]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # (B, maxk)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return tuple(res)


def build_model_from_config(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
   
    raise NotImplementedError(
        "Please implement `build_model_from_config` to create your backbone model."
    )


def build_dataloaders_from_config(
    cfg: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    
    raise NotImplementedError(
        "Please implement `build_dataloaders_from_config` to create your DataLoaders."
    )


def get_inputs_and_labels_from_batch(batch):
    
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, labels = batch
    elif isinstance(batch, dict):
        if "inputs" in batch and "labels" in batch:
            inputs = batch["inputs"]
            labels = batch["labels"]
        elif "video" in batch and "label" in batch:
            inputs = batch["video"]
            labels = batch["label"]
        else:
            raise KeyError(
                "Unsupported batch dict keys. Expected ('inputs','labels') or ('video','label')."
            )
    else:
        raise TypeError(
            "Unsupported batch type. Expected (inputs, labels) tuple or dict."
        )
    return inputs, labels


def train_one_epoch(
    model: nn.Module,
    criterion: LBORLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
) -> Dict[str, float]:
    model.train()

    batch_time = AverageMeter("Time")
    data_time = AverageMeter("Data")
    losses = AverageMeter("Loss")
    losses_ce = AverageMeter("CE")
    losses_lap = AverageMeter("Lap")
    losses_margin = AverageMeter("Margin")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, labels = get_inputs_and_labels_from_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        
        logits, features = model(inputs)

        total_loss, loss_dict = criterion(logits, features, labels)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(logits.detach(), labels, topk=(1, 5))

        batch_size = labels.size(0)
        losses.update(total_loss.item(), batch_size)
        losses_ce.update(loss_dict["loss_ce"].item(), batch_size)
        losses_lap.update(loss_dict["loss_lap"].item(), batch_size)
        losses_margin.update(loss_dict["loss_margin"].item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print(
                f"Epoch [{epoch}] "
                f"[{i + 1}/{len(train_loader)}] "
                f"{batch_time} {data_time} "
                f"{losses} CE:{losses_ce.avg:.4f} "
                f"Lap:{losses_lap.avg:.4f} Margin:{losses_margin.avg:.4f} "
                f"Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f}"
            )

    stats = {
        "loss": losses.avg,
        "loss_ce": losses_ce.avg,
        "loss_lap": losses_lap.avg,
        "loss_margin": losses_margin.avg,
        "acc1": top1.avg,
        "acc5": top5.avg,
    }
    return stats


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: LBORLoss,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
) -> Dict[str, float]:
    model.eval()

    batch_time = AverageMeter("Time")
    losses = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    end = time.time()

    for i, batch in enumerate(val_loader):
        inputs, labels = get_inputs_and_labels_from_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, features = model(inputs)
        total_loss, _ = criterion(logits, features, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = labels.size(0)
        losses.update(total_loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print(
                f"Validate Epoch [{epoch}] "
                f"[{i + 1}/{len(val_loader)}] "
                f"{batch_time} {losses} "
                f"Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f}"
            )

    stats = {
        "loss": losses.avg,
        "acc1": top1.avg,
        "acc5": top5.avg,
    }
    return stats


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Output directory
    output_dir = args.output_dir or cfg.get("misc", {}).get("output_dir", "outputs/default")
    os.makedirs(output_dir, exist_ok=True)

    # Seed
    seed = args.seed
    if seed is None:
        seed = int(cfg.get("misc", {}).get("seed", 42))
    set_random_seed(seed)
    print(f"[Info] Using random seed: {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")


    train_loader, val_loader = build_dataloaders_from_config(cfg)

    # Dataset / model related config
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})

    num_classes = int(dataset_cfg.get("num_classes"))
    feat_dim = int(model_cfg.get("feat_dim"))

   
    model = build_model_from_config(model_cfg, num_classes=num_classes)
    model.to(device)

    # LBOR loss
    criterion = LBORLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        lambda_lap=float(loss_cfg.get("lambda_lap", 1.0)),
        mu_margin=float(loss_cfg.get("mu_margin", 0.1)),
        margin_M=float(loss_cfg.get("margin_M", 1.0)),
        tau=float(loss_cfg.get("tau", 1.0)),
        use_knn=bool(loss_cfg.get("use_knn", False)),
        knn_k=int(loss_cfg.get("knn_k", 5)),
        center_momentum=float(loss_cfg.get("center_momentum", 0.1)),
    )
    criterion.to(device)

    # Optimizer
    base_lr = float(train_cfg.get("base_lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
    else:  # default: adamw
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    # Scheduler (optional)
    epochs = int(train_cfg.get("epochs", 100))
    sched_name = str(train_cfg.get("scheduler", "cosine")).lower()
    if sched_name == "none":
        scheduler = None
    elif sched_name == "step":
        step_size = int(train_cfg.get("step_size", 30))
        gamma = float(train_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:  # default cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    best_acc1 = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=int(train_cfg.get("print_freq", 50)),
        )

        val_stats = validate(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            print_freq=int(train_cfg.get("print_freq", 50)),
        )

        if scheduler is not None:
            scheduler.step()

        acc1 = val_stats["acc1"]
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch

        print(
            f"[Epoch {epoch}] "
            f"Train Loss {train_stats['loss']:.4f}, "
            f"Train Acc@1 {train_stats['acc1']:.2f}, "
            f"Val Loss {val_stats['loss']:.4f}, "
            f"Val Acc@1 {val_stats['acc1']:.2f}, "
            f"Best Acc@1 {best_acc1:.2f} (epoch {best_epoch})"
        )

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc1": best_acc1,
            "config": cfg,
        }
        torch.save(state, ckpt_path)
        if is_best:
            best_path = os.path.join(output_dir, "best.pth")
            torch.save(state, best_path)
            print(f"[Info] Saved new best checkpoint to {best_path}")

    print(f"\nTraining finished. Best Acc@1: {best_acc1:.2f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
