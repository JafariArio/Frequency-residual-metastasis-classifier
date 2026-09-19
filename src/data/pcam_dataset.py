from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple
import csv
import time

import h5py
import numpy as np
from torch.utils.data import Dataset

EXPECTED = {
    "train": {
        "x": "camelyonpatch_level_2_split_train_x.h5",
        "y": "camelyonpatch_level_2_split_train_y.h5",
        "meta": "camelyonpatch_level_2_split_train_meta.csv",
    },
    "valid": {
        "x": "camelyonpatch_level_2_split_valid_x.h5",
        "y": "camelyonpatch_level_2_split_valid_y.h5",
        "meta": "camelyonpatch_level_2_split_valid_meta.csv",
    },
    "test": {
        "x": "camelyonpatch_level_2_split_test_x.h5",
        "y": "camelyonpatch_level_2_split_test_y.h5",
        "meta": "camelyonpatch_level_2_split_test_meta.csv",
    },
}


def _resolve_real_file(root: Path, expected_name: str) -> Path:
    direct = root / expected_name
    if direct.is_file():
        return direct

    nested = root / expected_name / expected_name
    if nested.is_file():
        return nested

    matches = [p for p in root.rglob(expected_name) if p.is_file()]
    if matches:
        matches.sort(key=lambda p: (len(p.parts), str(p)))
        return matches[0]
    raise FileNotFoundError(f"Could not find real file for {expected_name} under {root}")


def _first_dataset_in_h5(h5: h5py.File):
    found = None

    def visitor(_name: str, obj: Any):
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset):
            found = obj

    h5.visititems(visitor)
    if found is None:
        raise RuntimeError("No dataset found inside HDF5 file")
    return found


class PCamH5Dataset(Dataset):
    """
    Windows/external-drive-safe PCam dataset.

    Main changes vs earlier version:
    - retries HDF5 reads on transient OSError/Invalid argument failures
    - closes and reopens file handles automatically on failed reads
    - opens with a larger raw-data chunk cache to reduce repeated small reads
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        return_meta: bool = False,
        read_retries: int = 3,
        retry_sleep_sec: float = 0.15,
    ) -> None:
        if split not in EXPECTED:
            raise ValueError(f"split must be one of {list(EXPECTED)}, got {split}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta
        self.read_retries = int(read_retries)
        self.retry_sleep_sec = float(retry_sleep_sec)

        names = EXPECTED[split]
        self.x_path = _resolve_real_file(self.root, names["x"])
        self.y_path = _resolve_real_file(self.root, names["y"])

        meta_candidate = self.root / names["meta"]
        self.meta_path: Optional[Path] = meta_candidate if meta_candidate.is_file() else None
        if self.meta_path is None:
            alt_meta = [p for p in self.root.rglob(names["meta"]) if p.is_file()]
            if alt_meta:
                alt_meta.sort(key=lambda p: (len(p.parts), str(p)))
                self.meta_path = alt_meta[0]

        self._x_h5: Optional[h5py.File] = None
        self._y_h5: Optional[h5py.File] = None
        self._x_ds = None
        self._y_ds = None
        self._meta_rows = self._load_meta() if self.return_meta else None
        self._length = self._infer_length()

    def _load_meta(self):
        if self.meta_path is None:
            return None
        rows = []
        with self.meta_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
        return rows

    def _open_h5(self, path: Path) -> h5py.File:
        # Larger chunk cache helps with repeated sample reads on big HDF5 files.
        return h5py.File(
            path,
            "r",
            rdcc_nbytes=64 * 1024 * 1024,
            rdcc_nslots=1_000_003,
        )

    def _ensure_open(self):
        if self._x_h5 is None:
            self._x_h5 = self._open_h5(self.x_path)
            self._x_ds = _first_dataset_in_h5(self._x_h5)
        if self._y_h5 is None:
            self._y_h5 = self._open_h5(self.y_path)
            self._y_ds = _first_dataset_in_h5(self._y_h5)

    def _reopen(self):
        self.close()
        self._ensure_open()

    def _infer_length(self) -> int:
        with self._open_h5(self.x_path) as f:
            x_ds = _first_dataset_in_h5(f)
            return int(x_ds.shape[0])

    def __len__(self) -> int:
        return self._length

    def _read_pair_once(self, index: int):
        self._ensure_open()
        image = np.array(self._x_ds[index])
        label_arr = np.array(self._y_ds[index])
        label = int(label_arr.squeeze())
        return image, label

    def __getitem__(self, index: int):
        last_err = None
        for attempt in range(self.read_retries + 1):
            try:
                image, label = self._read_pair_once(index)
                break
            except (OSError, ValueError) as e:
                last_err = e
                if attempt >= self.read_retries:
                    raise
                self._reopen()
                time.sleep(self.retry_sleep_sec)
        else:
            raise last_err  # pragma: no cover

        if self.transform is not None:
            image = self.transform(image)

        if self.return_meta:
            meta = self._meta_rows[index] if (self._meta_rows is not None and index < len(self._meta_rows)) else None
            return image, label, meta
        return image, label

    def close(self):
        if self._x_h5 is not None:
            try:
                self._x_h5.close()
            finally:
                self._x_h5 = None
                self._x_ds = None
        if self._y_h5 is not None:
            try:
                self._y_h5.close()
            finally:
                self._y_h5 = None
                self._y_ds = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def compute_mean_std(dataset: PCamH5Dataset, max_items: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(dataset), max_items)
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    count = 0
    for i in range(n):
        item = dataset[i]
        image = item[0] if isinstance(item, tuple) else item
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        image = image / 255.0 if image.max() > 1.5 else image
        pixels = image.reshape(-1, 3)
        sums += pixels.sum(axis=0)
        sq_sums += np.square(pixels).sum(axis=0)
        count += pixels.shape[0]
    mean = sums / count
    std = np.sqrt(np.maximum(sq_sums / count - np.square(mean), 0.0))
    return mean, std
