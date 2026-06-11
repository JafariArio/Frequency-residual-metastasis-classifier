#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from freqres_pathology.data import PCamH5Dataset, build_eval_transform
from freqres_pathology.models import FrequencyResidualClassifier, FrequencyResidualConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export positive-class probabilities from a trained checkpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    config = FrequencyResidualConfig(**config_dict)
    model = FrequencyResidualClassifier(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    dataset = PCamH5Dataset(args.data_root, args.split, transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device, non_blocking=True)
            scores = torch.sigmoid(model(images)["logits"]).cpu().numpy()
            labels = labels.cpu().numpy().astype(int)
            for i, (label, score) in enumerate(zip(labels, scores)):
                rows.append({"sample_id": offset + i, "y_true": int(label), "y_score": float(score)})
            offset += len(labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"rows": len(rows), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
