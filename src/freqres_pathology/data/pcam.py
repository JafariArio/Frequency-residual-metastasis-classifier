from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

PCAM_FILES = {
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


def _find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct
    nested = root / filename / filename
    if nested.is_file():
        return nested
    matches = sorted([p for p in root.rglob(filename) if p.is_file()], key=lambda p: (len(p.parts), str(p)))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def _first_dataset(handle: h5py.File):
    found = None

    def visit(_name: str, obj: Any):
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset):
            found = obj

    handle.visititems(visit)
    if found is None:
        raise RuntimeError("No dataset found in HDF5 file")
    return found


class PCamH5Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        return_meta: bool = False,
        read_retries: int = 3,
        retry_sleep_sec: float = 0.15,
    ) -> None:
        if split not in PCAM_FILES:
            raise ValueError(f"split must be one of {list(PCAM_FILES)}, got {split}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta
        self.read_retries = int(read_retries)
        self.retry_sleep_sec = float(retry_sleep_sec)

        files = PCAM_FILES[split]
        self.x_path = _find_file(self.root, files["x"])
        self.y_path = _find_file(self.root, files["y"])
        self.meta_path = self._find_optional_meta(files["meta"])

        self._x_h5 = None
        self._y_h5 = None
        self._x_ds = None
        self._y_ds = None
        self._meta_rows = self._load_meta() if self.return_meta else None
        self._length = self._infer_length()

    def _find_optional_meta(self, filename: str) -> Path | None:
        try:
            return _find_file(self.root, filename)
        except FileNotFoundError:
            return None

    def _load_meta(self):
        if self.meta_path is None:
            return None
        with self.meta_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _open_h5(path: Path) -> h5py.File:
        return h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)

    def _ensure_open(self) -> None:
        if self._x_h5 is None:
            self._x_h5 = self._open_h5(self.x_path)
            self._x_ds = _first_dataset(self._x_h5)
        if self._y_h5 is None:
            self._y_h5 = self._open_h5(self.y_path)
            self._y_ds = _first_dataset(self._y_h5)

    def _reopen(self) -> None:
        self.close()
        self._ensure_open()

    def _infer_length(self) -> int:
        with self._open_h5(self.x_path) as handle:
            return int(_first_dataset(handle).shape[0])

    def __len__(self) -> int:
        return self._length

    def _read_pair_once(self, index: int):
        self._ensure_open()
        image = np.array(self._x_ds[index])
        label = int(np.array(self._y_ds[index]).squeeze())
        return image, label

    def __getitem__(self, index: int):
        last_error = None
        for attempt in range(self.read_retries + 1):
            try:
                image, label = self._read_pair_once(index)
                break
            except (OSError, ValueError) as error:
                last_error = error
                if attempt >= self.read_retries:
                    raise
                self._reopen()
                time.sleep(self.retry_sleep_sec)
        else:
            raise last_error

        if self.transform is not None:
            image = self.transform(image)

        if self.return_meta:
            meta = self._meta_rows[index] if self._meta_rows is not None and index < len(self._meta_rows) else None
            return image, label, meta
        return image, label

    def close(self) -> None:
        if self._x_h5 is not None:
            self._x_h5.close()
            self._x_h5 = None
            self._x_ds = None
        if self._y_h5 is not None:
            self._y_h5.close()
            self._y_h5 = None
            self._y_ds = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def compute_mean_std(dataset: PCamH5Dataset, max_items: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    n_items = min(len(dataset), int(max_items))
    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    count = 0

    for index in range(n_items):
        item = dataset[index]
        image = item[0] if isinstance(item, tuple) else item
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        image = image / 255.0 if image.max() > 1.5 else image
        pixels = image.reshape(-1, 3)
        sums += pixels.sum(axis=0)
        squared_sums += np.square(pixels).sum(axis=0)
        count += pixels.shape[0]

    mean = sums / count
    std = np.sqrt(np.maximum(squared_sums / count - np.square(mean), 0.0))
    return mean, std
