#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from freqres_pathology.data import PCamH5Dataset, build_eval_transform, build_train_transform
from freqres_pathology.eval.metrics import binary_metrics


def build_model(name: str):
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    try:
        import timm
    except ImportError as error:
        raise ImportError("Install timm to train non-ResNet baselines.") from error
    return timm.create_model(name, pretrained=False, num_classes=1)


def forward_logits(model, images):
    logits = model(images)
    return logits.squeeze(1) if logits.ndim > 1 else logits


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_n = 0
    rows = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = forward_logits(model, images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        rows.append(pd.DataFrame({"y_true": labels.detach().cpu().int().numpy(), "y_score": scores}))
        total_loss += float(loss.detach().cpu()) * len(labels)
        total_n += len(labels)
    frame = pd.concat(rows, ignore_index=True)
    metrics = binary_metrics(frame["y_true"], frame["y_score"], threshold=0.5)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline image classifier on PCam.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_set = PCamH5Dataset(args.data_root, "train", transform=build_train_transform(use_translation=True))
    valid_set = PCamH5Dataset(args.data_root, "valid", transform=build_eval_transform())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if valid_metrics["balanced_accuracy"] > best_score:
            best_score = valid_metrics["balanced_accuracy"]
            torch.save({"model_state": model.state_dict(), "model_name": args.model}, out_dir / "best_checkpoint.pt")
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
