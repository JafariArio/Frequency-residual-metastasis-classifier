#!/usr/bin/env python
r"""
Phase 13C — Stream and save the full CAMELYON17-WILDS test split.

Purpose
-------
Create a local full external-transfer test set for the PathEOM / MetaPath Paper 1
project without downloading the full CAMELYON17-WILDS train/validation/test corpus.

Dataset source
--------------
Hugging Face dataset:
  wltjr1007/Camelyon17-WILDS

Split:
  test

This script streams the entire test split to completion and saves every test row.

Default output
--------------
  <repository>/data/raw/wilds_full_test

Files written
-------------
- camelyon17_wilds_full_test_x.h5
- camelyon17_wilds_full_test_y.h5
- camelyon17_wilds_full_test_metadata.csv
- wilds_full_test_manifest.json

HDF5 structure
--------------
x.h5:
  dataset key "x"
  shape (N, 96, 96, 3)
  dtype uint8

y.h5:
  dataset key "y"
  shape (N, 1, 1, 1)
  dtype uint8

Scientific role
---------------
This is the full external test split for a frozen PCam-trained PathEOM transfer
evaluation. No balancing or class equalization is applied here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception as exc:
    raise RuntimeError(
        "The 'datasets' package is required. Install it with: py -m pip install datasets"
    ) from exc


DATASET_NAME = "wltjr1007/Camelyon17-WILDS"
SPLIT_NAME = "test"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and save the full CAMELYON17-WILDS test split."
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "wilds_full_test"),
        help="Directory where full-test HDF5 files and metadata will be written.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "hf_cache"),
        help="Hugging Face datasets cache directory on the E: drive.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print status every N streamed records.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression for saved image/label datasets.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level when --compression gzip is used.",
    )
    parser.add_argument(
        "--append-block-size",
        type=int,
        default=256,
        help="Number of streamed examples buffered before one HDF5 append write.",
    )
    parser.add_argument(
        "--max-stream-records",
        type=int,
        default=0,
        help="Optional debugging cap. 0 means stream the full test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def compression_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": int(args.compression_level)}


def ensure_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "outdir": outdir,
        "x_h5": outdir / "camelyon17_wilds_full_test_x.h5",
        "y_h5": outdir / "camelyon17_wilds_full_test_y.h5",
        "metadata_csv": outdir / "camelyon17_wilds_full_test_metadata.csv",
        "manifest_json": outdir / "wilds_full_test_manifest.json",
    }
    if not args.overwrite:
        existing = [str(p) for k, p in paths.items() if k != "outdir" and p.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use a new --outdir or pass --overwrite.\n"
                + "\n".join(existing)
            )
    return paths


def image_to_uint8_rgb(image_obj: Any) -> np.ndarray:
    if isinstance(image_obj, Image.Image):
        img = image_obj.convert("RGB")
    elif isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            img = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        elif image_obj.get("path") is not None:
            img = Image.open(image_obj["path"]).convert("RGB")
        else:
            raise TypeError(f"Unsupported image dict payload keys: {list(image_obj.keys())}")
    elif isinstance(image_obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_obj)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image object type: {type(image_obj)!r}")

    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape {arr.shape}")
    if arr.shape[0] != 96 or arr.shape[1] != 96:
        img = img.resize((96, 96), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(arr)


def label_to_int(label_obj: Any) -> int:
    if isinstance(label_obj, str):
        normalized = label_obj.strip().lower()
        if normalized in {"0", "non-tumor", "non_tumor", "negative", "normal"}:
            return 0
        if normalized in {"1", "tumor", "positive"}:
            return 1
        raise ValueError(f"Unrecognized string label: {label_obj!r}")
    label = int(label_obj)
    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {label}")
    return label


def metadata_value(row: Dict[str, Any], key: str) -> Any:
    value = row.get(key, "")
    if isinstance(value, np.generic):
        return value.item()
    return value


def initialize_h5_files(paths: Dict[str, Path], args: argparse.Namespace):
    comp = compression_kwargs(args)
    block = max(1, int(args.append_block_size))
    x_h5 = h5py.File(paths["x_h5"], "w")
    y_h5 = h5py.File(paths["y_h5"], "w")
    x_ds = x_h5.create_dataset(
        "x",
        shape=(0, 96, 96, 3),
        maxshape=(None, 96, 96, 3),
        chunks=(min(block, 256), 96, 96, 3),
        dtype=np.uint8,
        **comp,
    )
    y_ds = y_h5.create_dataset(
        "y",
        shape=(0, 1, 1, 1),
        maxshape=(None, 1, 1, 1),
        chunks=(min(max(block, 1), 4096), 1, 1, 1),
        dtype=np.uint8,
        **comp,
    )
    return x_h5, y_h5, x_ds, y_ds


def flush_block(
    x_ds,
    y_ds,
    image_buffer: List[np.ndarray],
    label_buffer: List[int],
) -> int:
    if not image_buffer:
        return 0
    start = int(x_ds.shape[0])
    n = len(image_buffer)
    end = start + n
    x_ds.resize((end, 96, 96, 3))
    y_ds.resize((end, 1, 1, 1))
    x_ds[start:end] = np.stack(image_buffer, axis=0)
    labels = np.asarray(label_buffer, dtype=np.uint8).reshape(n, 1, 1, 1)
    y_ds[start:end] = labels
    return n


def main() -> int:
    args = parse_args()
    if args.append_block_size <= 0:
        raise ValueError("--append-block-size must be positive.")

    paths = ensure_output_paths(args)
    hf_cache_dir = Path(args.hf_cache_dir)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_cache_dir.parent / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir.parent / "hf_hub_cache"))

    status(f"Loading streaming dataset: {DATASET_NAME}, split={SPLIT_NAME}")
    stream = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
        streaming=True,
        cache_dir=str(hf_cache_dir),
    )

    counts = {0: 0, 1: 0}
    streamed_records = 0
    saved_records = 0
    start_time = time.time()

    fieldnames = [
        "subset_index",
        "stream_index",
        "label",
        "center",
        "image_id",
        "patient",
        "node",
        "x_coord",
        "y_coord",
        "slide",
    ]

    x_h5 = y_h5 = x_ds = y_ds = None
    metadata_file = None
    image_buffer: List[np.ndarray] = []
    label_buffer: List[int] = []
    metadata_buffer: List[Dict[str, Any]] = []

    try:
        x_h5, y_h5, x_ds, y_ds = initialize_h5_files(paths, args)
        metadata_file = paths["metadata_csv"].open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(desc="Streaming full CAMELYON17-WILDS test split", unit="row")
        for stream_index, row in enumerate(stream):
            streamed_records += 1
            pbar.update(1)

            if args.max_stream_records > 0 and streamed_records > args.max_stream_records:
                status(f"Reached --max-stream-records={args.max_stream_records}; stopping.")
                break

            label = label_to_int(row["label"])
            image_arr = image_to_uint8_rgb(row["image"])
            future_subset_index = saved_records + len(image_buffer)
            image_buffer.append(image_arr)
            label_buffer.append(label)
            metadata_buffer.append(
                {
                    "subset_index": future_subset_index,
                    "stream_index": int(stream_index),
                    "label": int(label),
                    "center": metadata_value(row, "center"),
                    "image_id": metadata_value(row, "image_id"),
                    "patient": metadata_value(row, "patient"),
                    "node": metadata_value(row, "node"),
                    "x_coord": metadata_value(row, "x_coord"),
                    "y_coord": metadata_value(row, "y_coord"),
                    "slide": metadata_value(row, "slide"),
                }
            )
            counts[label] += 1

            if len(image_buffer) >= args.append_block_size:
                wrote = flush_block(x_ds, y_ds, image_buffer, label_buffer)
                for meta_row in metadata_buffer:
                    writer.writerow(meta_row)
                saved_records += wrote
                image_buffer.clear()
                label_buffer.clear()
                metadata_buffer.clear()

            if args.progress_every > 0 and streamed_records % args.progress_every == 0:
                status(
                    f"streamed={streamed_records:,}; saved={saved_records + len(image_buffer):,}; "
                    f"negative={counts[0]:,}; positive={counts[1]:,}"
                )

        if image_buffer:
            wrote = flush_block(x_ds, y_ds, image_buffer, label_buffer)
            for meta_row in metadata_buffer:
                writer.writerow(meta_row)
            saved_records += wrote
            image_buffer.clear()
            label_buffer.clear()
            metadata_buffer.clear()

        pbar.close()
        x_h5.flush()
        y_h5.flush()
        metadata_file.flush()

    except Exception as exc:
        failure = {
            "phase": "Phase 13C full CAMELYON17-WILDS test split download",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "streamed_records_before_failure": streamed_records,
            "saved_records_before_failure": saved_records,
            "counts_before_failure": counts,
        }
        write_json(paths["outdir"] / "FAILED_wilds_full_test_download_error.json", failure)
        raise
    finally:
        try:
            if metadata_file is not None:
                metadata_file.close()
        except Exception:
            pass
        try:
            if x_h5 is not None:
                x_h5.close()
        except Exception:
            pass
        try:
            if y_h5 is not None:
                y_h5.close()
        except Exception:
            pass

    elapsed_sec = time.time() - start_time
    manifest = {
        "phase": "Phase 13C CAMELYON17-WILDS full streamed test split",
        "status": "COMPLETED",
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "selection_rule": "stream every row in the Hugging Face test split to completion; no balancing or subsampling",
        "saved_total": int(saved_records),
        "saved_negative_label0": int(counts[0]),
        "saved_positive_label1": int(counts[1]),
        "streamed_records_scanned": int(streamed_records),
        "elapsed_seconds": float(elapsed_sec),
        "paths": {k: str(v) for k, v in paths.items()},
        "hf_cache_dir": str(hf_cache_dir),
        "environment": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
        "hdf5_structure": {
            "x_file": {"key": "x", "shape": [int(saved_records), 96, 96, 3], "dtype": "uint8"},
            "y_file": {"key": "y", "shape": [int(saved_records), 1, 1, 1], "dtype": "uint8"},
        },
    }
    write_json(paths["manifest_json"], manifest)
    status(
        f"Completed full test split. saved={saved_records:,}; negative={counts[0]:,}; "
        f"positive={counts[1]:,}; streamed={streamed_records:,}"
    )
    print(json.dumps(jsonable(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

