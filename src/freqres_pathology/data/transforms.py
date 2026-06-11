from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, float, float] = (0.700, 0.540, 0.690)
    std: tuple[float, float, float] = (0.235, 0.275, 0.212)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class ToTensor:
    def __call__(self, image):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 image array, got shape {array.shape}")
        array = array.astype(np.float32) / (255.0 if array.max() > 1.5 else 1.0)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class Normalize:
    def __init__(self, stats: NormalizationStats):
        self.mean = torch.tensor(stats.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(stats.std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std.clamp_min(1e-8)


class RandomHorizontalVerticalFlip:
    def __call__(self, image):
        array = np.asarray(image)
        if np.random.rand() < 0.5:
            array = np.flip(array, axis=1)
        if np.random.rand() < 0.5:
            array = np.flip(array, axis=0)
        return array.copy()


class RandomTranslation:
    def __init__(self, max_shift: int = 4):
        self.max_shift = int(max_shift)

    def __call__(self, image):
        array = np.asarray(image)
        if self.max_shift <= 0:
            return array
        dy = np.random.randint(-self.max_shift, self.max_shift + 1)
        dx = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(np.roll(array, dy, axis=0), dx, axis=1).copy()


def build_train_transform(stats: NormalizationStats | None = None, use_translation: bool = False):
    stats = stats or NormalizationStats()
    transforms = [RandomHorizontalVerticalFlip()]
    if use_translation:
        transforms.append(RandomTranslation(max_shift=4))
    transforms.extend([ToTensor(), Normalize(stats)])
    return Compose(transforms)


def build_eval_transform(stats: NormalizationStats | None = None):
    stats = stats or NormalizationStats()
    return Compose([ToTensor(), Normalize(stats)])
