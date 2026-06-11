from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrequencyResidualConfig:
    in_channels: int = 3
    stem_dim: int = 64
    embed_dim: int = 192
    depth: int = 6
    mlp_ratio: float = 2.0
    scales: tuple[int, ...] = (4, 8, 16)
    use_multiband_filter: bool = False
    quality_head: bool = True
    dropout: float = 0.0


class ConvNormActivation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MorphologyStem(nn.Module):
    def __init__(self, in_channels: int, stem_dim: int):
        super().__init__()
        self.stage1 = ConvNormActivation(in_channels, stem_dim // 2, kernel_size=3, stride=2)
        self.stage2 = ConvNormActivation(stem_dim // 2, stem_dim, kernel_size=3, stride=2)
        self.local = nn.Sequential(
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, padding=1, groups=stem_dim, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return self.activation(x + self.local(x))


class ScaleTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, scales: Iterable[int]):
        super().__init__()
        self.scales = tuple(scales)
        self.projections = nn.ModuleDict(
            {str(scale): nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False) for scale in self.scales}
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        tokens = []
        for scale in self.scales:
            pooled = F.adaptive_avg_pool2d(features, output_size=(scale, scale))
            tokens.append(self.projections[str(scale)](pooled))
        return tokens


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, embed_dim: int, n_scales: int, target_size: int = 8):
        super().__init__()
        hidden = max(embed_dim // 2, 32)
        self.target_size = target_size
        self.scale_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_scales),
        )
        self.projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        resized = [
            F.interpolate(z, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
            for z in scale_features
        ]
        pooled = torch.stack([z.mean(dim=(2, 3)) for z in resized], dim=1)
        weights = torch.softmax(self.scale_gate(pooled.mean(dim=1)), dim=1)
        fused = 0.0
        for i, z in enumerate(resized):
            fused = fused + z * weights[:, i].view(-1, 1, 1, 1)
        return self.projection(fused)


class ChannelMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class FrequencyResidualOperator(nn.Module):
    def __init__(self, dim: int, use_multiband: bool = False):
        super().__init__()
        self.use_multiband = bool(use_multiband)
        n_bands = 3 if self.use_multiband else 1
        hidden = max(dim // 4, 16)
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_bands))
        self.band_weights = nn.Parameter(torch.ones(n_bands, dim))

    @staticmethod
    def _band_masks(height: int, width_rfft: int, device: torch.device) -> torch.Tensor:
        yy = torch.arange(height, device=device).view(height, 1).float()
        xx = torch.arange(width_rfft, device=device).view(1, width_rfft).float()
        radius = torch.sqrt((yy / max(height - 1, 1)) ** 2 + (xx / max(width_rfft - 1, 1)) ** 2)
        low = (radius <= 0.35).float()
        mid = ((radius > 0.20) & (radius <= 0.70)).float()
        high = (radius > 0.55).float()
        return torch.stack([low, mid, high], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        fft = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        masks = self._band_masks(height, fft.shape[-1], x.device)
        gate = torch.sigmoid(self.gate(x.mean(dim=(2, 3))))

        if self.use_multiband:
            filtered = torch.zeros_like(fft)
            for i in range(3):
                mask = masks[i].view(1, 1, height, fft.shape[-1])
                weight = self.band_weights[i].view(1, channels, 1, 1) * gate[:, i].view(batch, 1, 1, 1)
                filtered = filtered + fft * mask * weight
        else:
            mask = masks[1].view(1, 1, height, fft.shape[-1])
            weight = self.band_weights[0].view(1, channels, 1, 1) * gate[:, 0].view(batch, 1, 1, 1)
            filtered = fft * mask * weight

        y = torch.fft.irfft2(filtered, s=(height, width), dim=(-2, -1), norm="ortho")
        return y.type_as(x)


class FrequencyResidualBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, use_multiband_filter: bool = False, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ChannelMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.norm3 = nn.BatchNorm2d(dim)
        self.frequency = FrequencyResidualOperator(dim, use_multiband=use_multiband_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x + self.frequency(self.norm3(x))
        return x


class FrequencyResidualClassifier(nn.Module):
    def __init__(self, config: FrequencyResidualConfig):
        super().__init__()
        self.config = config
        self.stem = MorphologyStem(config.in_channels, config.stem_dim)
        self.tokenizer = ScaleTokenizer(config.stem_dim, config.embed_dim, config.scales)
        self.fusion = AdaptiveScaleFusion(config.embed_dim, len(config.scales), target_size=8)
        self.blocks = nn.Sequential(
            *[
                FrequencyResidualBlock(
                    dim=config.embed_dim,
                    mlp_ratio=config.mlp_ratio,
                    use_multiband_filter=config.use_multiband_filter,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.head_norm = nn.BatchNorm1d(config.embed_dim * 2)
        self.classifier = nn.Linear(config.embed_dim * 2, 1)
        self.quality_head = nn.Linear(config.embed_dim * 2, 1) if config.quality_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        tokens = self.tokenizer(features)
        fused = self.fusion(tokens)
        return self.blocks(fused)

    def pooled_summary(self, token_grid: torch.Tensor) -> torch.Tensor:
        mean_pool = token_grid.mean(dim=(2, 3))
        max_pool = token_grid.amax(dim=(2, 3))
        return self.head_norm(torch.cat([mean_pool, max_pool], dim=1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        token_grid = self.forward_features(x)
        features = self.pooled_summary(token_grid)
        logits = self.classifier(features).squeeze(1)
        output = {"logits": logits, "features": features, "token_grid": token_grid}
        if self.quality_head is not None:
            output["quality_logits"] = self.quality_head(features).squeeze(1)
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
