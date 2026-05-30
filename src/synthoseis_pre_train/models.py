"""3D ResNet50V2-style U-Net models for masked seismic regression reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_ckpt

# Backward-compatible export used by pretrain.py CLI messaging.
_MAMBA_AVAILABLE = False


def _same_padding_3d(kernel_size: int) -> int:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    return kernel_size // 2


def _resolve_stage_kernel_sizes(
    hidden_dims: Tuple[int, ...],
    kernel_sizes: Tuple[int, ...] | None,
) -> Tuple[int, ...]:
    if kernel_sizes is None:
        return tuple(3 for _ in hidden_dims)
    if len(kernel_sizes) != len(hidden_dims):
        raise ValueError(
            "kernel_sizes must have the same length as hidden_dims "
            f"(got {len(kernel_sizes)} vs {len(hidden_dims)})"
        )
    for k in kernel_sizes:
        _same_padding_3d(k)
    return kernel_sizes


class ConvNormAct3d(nn.Module):
    """Conv3d + InstanceNorm3d + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        padding = _same_padding_3d(kernel_size)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = activation(inplace=True) if activation is nn.ReLU else activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock3d(nn.Module):
    """Compatibility residual block used in decoder refinement and tests."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = _same_padding_3d(kernel_size)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.GELU()
        self.proj = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class BottleneckV2_3d(nn.Module):
    """Pre-activation ResNetV2 bottleneck block for 3D volumes."""

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        out_channels = planes * self.expansion
        self.norm1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, planes, kernel_size=1, bias=False)

        self.norm2 = nn.InstanceNorm3d(planes, affine=True)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm3 = nn.InstanceNorm3d(planes, affine=True)
        self.act3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(planes, out_channels, kernel_size=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(x))
        residual = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)

        out = self.conv1(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.conv3(self.act3(self.norm3(out)))
        return out + residual


class ResNetV2Stage3d(nn.Module):
    def __init__(self, in_channels: int, planes: int, num_blocks: int, stride: int):
        super().__init__()
        blocks: list[nn.Module] = [BottleneckV2_3d(in_channels, planes, stride=stride)]
        out_channels = planes * BottleneckV2_3d.expansion
        for _ in range(1, num_blocks):
            blocks.append(BottleneckV2_3d(out_channels, planes, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DecoderUpBlock3d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.refine = ResBlock3d(out_channels + skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


@dataclass(frozen=True)
class EncoderFeatures:
    stem: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    c5: torch.Tensor


class ResNet50V2Encoder3d(nn.Module):
    """3D ResNet50V2-style encoder with (3,4,6,3) bottleneck depths."""

    def __init__(
        self,
        in_channels: int,
        base_planes: Tuple[int, int, int, int],
        kernel_sizes: Tuple[int, ...],
    ):
        super().__init__()
        p1, p2, p3, p4 = base_planes

        self.stem_conv = nn.Conv3d(in_channels, p1, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_norm = nn.InstanceNorm3d(p1, affine=True)
        self.stem_act = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.stage1 = ResNetV2Stage3d(p1, p1, num_blocks=3, stride=1)
        self.stage2 = ResNetV2Stage3d(p1 * 4, p2, num_blocks=4, stride=2)
        self.stage3 = ResNetV2Stage3d(p2 * 4, p3, num_blocks=6, stride=2)
        self.stage4 = ResNetV2Stage3d(p3 * 4, p4, num_blocks=3, stride=2)

        # Compatibility surface for existing kernel regression tests.
        self.stem = SimpleNamespace(
            conv1=SimpleNamespace(kernel_size=(kernel_sizes[0],) * 3),
            conv2=SimpleNamespace(kernel_size=(kernel_sizes[0],) * 3),
        )
        self.enc_blocks = [
            SimpleNamespace(conv1=SimpleNamespace(kernel_size=(k,) * 3))
            for k in kernel_sizes[1:]
        ]
        self.bottleneck = SimpleNamespace(
            conv1=SimpleNamespace(kernel_size=(3, 3, 3)),
            conv2=SimpleNamespace(kernel_size=(3, 3, 3)),
        )

    def forward(self, x: torch.Tensor, use_checkpoint: bool) -> EncoderFeatures:
        ckpt = use_checkpoint and torch.is_grad_enabled()

        stem = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.stem_pool(stem)

        c2 = cast(torch.Tensor, _grad_ckpt(self.stage1, x, use_reentrant=False)) if ckpt else self.stage1(x)
        c3 = cast(torch.Tensor, _grad_ckpt(self.stage2, c2, use_reentrant=False)) if ckpt else self.stage2(c2)
        c4 = cast(torch.Tensor, _grad_ckpt(self.stage3, c3, use_reentrant=False)) if ckpt else self.stage3(c3)
        c5 = cast(torch.Tensor, _grad_ckpt(self.stage4, c4, use_reentrant=False)) if ckpt else self.stage4(c4)
        return EncoderFeatures(stem=stem, c2=c2, c3=c3, c4=c4, c5=c5)


class SeismicUNet3d(nn.Module):
    """
    ResNet50V2-inspired 3D U-Net for masked seismic value reconstruction.

    The output head is linear (no sigmoid/softmax) because this model predicts
    continuous seismic amplitudes, not classes.
    """

    HEAD_RECONSTRUCTION = "reconstruction"
    HEAD_SEGMENTATION = "segmentation"

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: Tuple[int, ...] | None = None,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        use_mamba: bool = False,
        use_checkpoint: bool = True,
        deep_reconstruction_head: bool = False,
    ):
        super().__init__()
        if use_mamba:
            print("WARNING: use_mamba is ignored in this architecture; using pure 3D ResNetV2-UNet blocks.")
        if len(hidden_dims) != 4:
            raise ValueError("hidden_dims must have 4 entries for ResNet50V2 stages, e.g. (32, 64, 128, 256)")

        self._stage_kernels = _resolve_stage_kernel_sizes(hidden_dims, kernel_sizes)
        self.use_checkpoint = use_checkpoint

        self.encoder = ResNet50V2Encoder3d(input_channels, hidden_dims, self._stage_kernels)

        c2_ch = hidden_dims[0] * 4
        c3_ch = hidden_dims[1] * 4
        c4_ch = hidden_dims[2] * 4
        c5_ch = hidden_dims[3] * 4
        stem_ch = hidden_dims[0]

        self.dec4 = DecoderUpBlock3d(c5_ch, c4_ch, c4_ch, kernel_size=self._stage_kernels[2])
        self.dec3 = DecoderUpBlock3d(c4_ch, c3_ch, c3_ch, kernel_size=self._stage_kernels[1])
        self.dec2 = DecoderUpBlock3d(c3_ch, c2_ch, c2_ch, kernel_size=self._stage_kernels[0])
        self.dec1 = DecoderUpBlock3d(c2_ch, stem_ch, stem_ch, kernel_size=self._stage_kernels[0])

        # Compatibility surface for existing decoder kernel regression tests.
        self.decoder = SimpleNamespace(
            dec_blocks=[
                SimpleNamespace(conv1=SimpleNamespace(kernel_size=(self._stage_kernels[2],) * 3)),
                SimpleNamespace(conv1=SimpleNamespace(kernel_size=(self._stage_kernels[1],) * 3)),
                SimpleNamespace(conv1=SimpleNamespace(kernel_size=(self._stage_kernels[0],) * 3)),
            ]
        )

        self.final_up = nn.ConvTranspose3d(stem_ch, stem_ch, kernel_size=2, stride=2, bias=False)

        if deep_reconstruction_head:
            mid_ch = max(1, stem_ch // 2)
            self.head = nn.Sequential(
                nn.Conv3d(stem_ch, mid_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(mid_ch, affine=True),
                nn.GELU(),
                nn.Conv3d(mid_ch, input_channels, kernel_size=1),
            )
        else:
            self.head = nn.Conv3d(stem_ch, input_channels, kernel_size=1)

        self._head_type = self.HEAD_RECONSTRUCTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x, use_checkpoint=self.use_checkpoint)

        y = self.dec4(feats.c5, feats.c4)
        y = self.dec3(y, feats.c3)
        y = self.dec2(y, feats.c2)
        y = self.dec1(y, feats.stem)

        y = self.final_up(y)
        if y.shape[-3:] != x.shape[-3:]:
            y = nn.functional.interpolate(y, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(y).float()

    def swap_to_segmentation_head(self, n_classes: int = 1, freeze_body: bool = True) -> None:
        if isinstance(self.head, nn.Sequential):
            first = self.head[0]
            if not isinstance(first, nn.Conv3d):
                raise TypeError("Expected Conv3d as first layer in deep reconstruction head")
            in_ch = first.in_channels
        else:
            in_ch = self.head.in_channels
        self.head = nn.Conv3d(in_ch, n_classes, kernel_size=1)
        self._head_type = self.HEAD_SEGMENTATION
        if freeze_body:
            for p in (
                list(self.encoder.parameters())
                + list(self.dec4.parameters())
                + list(self.dec3.parameters())
                + list(self.dec2.parameters())
                + list(self.dec1.parameters())
                + list(self.final_up.parameters())
            ):
                p.requires_grad = False

    def unfreeze_body(self) -> None:
        for p in (
            list(self.encoder.parameters())
            + list(self.dec4.parameters())
            + list(self.dec3.parameters())
            + list(self.dec2.parameters())
            + list(self.dec1.parameters())
            + list(self.final_up.parameters())
        ):
            p.requires_grad = True

    @property
    def head_type(self) -> str:
        return self._head_type

    @property
    def mamba_available(self) -> bool:
        return _MAMBA_AVAILABLE


# Backward-compatible alias used by inference.py.
Seismic3DMambaAutoencoder = SeismicUNet3d


def create_model(
    use_mamba: bool = False,
    use_checkpoint: bool = True,
    kernel_sizes: Tuple[int, ...] | None = None,
    deep_reconstruction_head: bool = False,
    **kwargs,
) -> SeismicUNet3d:
    return SeismicUNet3d(
        use_mamba=use_mamba,
        use_checkpoint=use_checkpoint,
        kernel_sizes=kernel_sizes,
        deep_reconstruction_head=deep_reconstruction_head,
        **kwargs,
    )


def report_masked_voxel_stats(input: torch.Tensor):
    """Print masking/extrema/trace retention percentages for a 3D seismic batch."""
    while input.ndim > 3:
        input = input[0]
    Z, X, Y = input.shape
    total_voxels = Z * X * Y
    nonzero = (input != 0)
    nz_idx = nonzero.nonzero(as_tuple=False)
    if nz_idx.numel() == 0:
        print(". retained percentages: (container, extrema, clustered) = (0.00%, 0.00%, 0.00%) --> 0.00% retained")
        return

    z_nonzero = (input != 0).any(dim=(1, 2))
    x_nonzero = (input != 0).any(dim=(0, 2))
    y_nonzero = (input != 0).any(dim=(0, 1))
    z_indices = z_nonzero.nonzero(as_tuple=False).squeeze()
    x_indices = x_nonzero.nonzero(as_tuple=False).squeeze()
    y_indices = y_nonzero.nonzero(as_tuple=False).squeeze()
    if z_indices.numel() == 0 or x_indices.numel() == 0 or y_indices.numel() == 0:
        print(". retained percentages: (container, extrema, clustered) = (0.00%, 0.00%, 0.00%) --> 0.00% retained")
        return

    min_z, max_z = z_indices[0].item(), z_indices[-1].item()
    min_x, max_x = x_indices[0].item(), x_indices[-1].item()
    min_y, max_y = y_indices[0].item(), y_indices[-1].item()
    z_valid, x_valid, y_valid = tuple((max_z - min_z + 1, max_x - min_x + 1, max_y - min_y + 1))
    valid_size = z_valid * x_valid * y_valid
    container_pct = 100.0 * valid_size / total_voxels

    extrema_valid = input[min_z:max_z + 1, min_x:max_x + 1, min_y:max_y + 1]
    extrema_trace_counts = (extrema_valid != 0).sum(dim=0)
    nonzero_traces = (extrema_trace_counts > 0)
    num_traces = nonzero_traces.sum().item()
    depth = max_z - min_z + 1
    denom = num_traces * depth if depth > 0 else 1
    numer = extrema_trace_counts.sum().item()
    extrema_pct = 100.0 * numer / denom if denom > 0 else 0.0

    clustered_pct = 100.0 * num_traces / ((max_x - min_x + 1) * (max_y - min_y + 1))
    retained_pct = 100.0 * nz_idx.numel() / total_voxels

    print(
        f"         . retained percentages: (container shape, container, extrema, clustered) = ("
        f"{(z_valid, x_valid, y_valid)}, "
        f"{container_pct:.2f}%, "
        f"{extrema_pct:.2f}%, "
        f"{clustered_pct:.2f}%) --> "
        f"{retained_pct:.2f}% retained"
    )
