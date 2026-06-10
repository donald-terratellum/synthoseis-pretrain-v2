"""3D ResNet50V2-style U-Net models for masked seismic regression reconstruction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_ckpt

# Backward-compatible export used by pretrain.py CLI messaging.
_MAMBA_AVAILABLE = False


def _same_padding_3d(kernel_size: int) -> int:
    """Return symmetric padding for odd 3D kernels at stride 1.

    Args:
        kernel_size: Convolution kernel size.

    Returns:
        The integer padding value that preserves spatial shape for stride-1 convs.

    Raises:
        ValueError: If ``kernel_size`` is non-positive or even.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    return kernel_size // 2


def _resolve_stage_kernel_sizes(
    hidden_dims: Tuple[int, ...],
    kernel_sizes: Tuple[int, ...] | None,
) -> Tuple[int, ...]:
    """Resolve per-stage kernel sizes for encoder/decoder refinement blocks.

    Args:
        hidden_dims: Channel widths defining model stages.
        kernel_sizes: Optional odd kernel size per stage.

    Returns:
        A kernel-size tuple with one entry per stage.

    Raises:
        ValueError: If ``kernel_sizes`` length mismatches ``hidden_dims`` or any
            kernel value is invalid.
    """
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
        """Initialize a 3D conv-norm-activation block.

        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            kernel_size: Odd convolution kernel size.
            stride: Convolution stride.
            activation: Activation module class.
        """
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
        """Apply convolution, normalization, and activation."""
        return self.act(self.norm(self.conv(x)))


class ResBlock3d(nn.Module):
    """Compatibility residual block used in decoder refinement and tests."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialize a two-convolution residual block.

        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            kernel_size: Odd kernel size used by both convolutions.
        """
        super().__init__()
        padding = _same_padding_3d(kernel_size)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.GELU()
        self.proj = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual refinement output with GELU activation."""
        residual = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class BottleneckV2_3d(nn.Module):
    """Pre-activation ResNetV2 bottleneck block for 3D volumes."""

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        """Initialize a 3D ResNetV2 bottleneck block.

        Args:
            in_channels: Input feature channels.
            planes: Bottleneck inner-channel width before expansion.
            stride: Spatial stride for the 3x3 bottleneck convolution.
        """
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
        """Run pre-activation bottleneck transform and residual addition."""
        out = self.act1(self.norm1(x))
        residual = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)

        out = self.conv1(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.conv3(self.act3(self.norm3(out)))
        return out + residual


class ResNetV2Stage3d(nn.Module):
    """A sequence of ResNetV2 bottleneck blocks at one feature scale."""

    def __init__(self, in_channels: int, planes: int, num_blocks: int, stride: int):
        """Initialize one ResNetV2 stage.

        Args:
            in_channels: Stage input channels.
            planes: Bottleneck base channels.
            num_blocks: Number of bottleneck blocks in the stage.
            stride: Stride used by the first block.
        """
        super().__init__()
        blocks: list[nn.Module] = [BottleneckV2_3d(in_channels, planes, stride=stride)]
        out_channels = planes * BottleneckV2_3d.expansion
        for _ in range(1, num_blocks):
            blocks.append(BottleneckV2_3d(out_channels, planes, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all bottleneck blocks in this stage."""
        return self.blocks(x)


class DecoderUpBlock3d(nn.Module):
    """Resize-conv upsample + skip-concat + residual refinement block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int):
        """Initialize one decoder upsampling block.

        Args:
            in_channels: Input channels from deeper decoder level.
            skip_channels: Channels in the matching encoder skip tensor.
            out_channels: Output channels after upsampling/refinement.
            kernel_size: Odd kernel size for residual refinement convolutions.
        """
        super().__init__()
        # Resize-convolution avoids checkerboard artifacts from transpose conv overlap.
        self.up_interp = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.refine = ResBlock3d(out_channels + skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample input, align spatially with skip, then fuse and refine."""
        x = self.up_interp(x)
        x = self.up_conv(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = nn.functional.interpolate(x, size=skip.shape[-3:], mode="nearest")
        return self.refine(torch.cat([x, skip], dim=1))


class ResNet50V2Encoder3d(nn.Module):
    """3D ResNet50V2-style encoder with (3,4,6,3) bottleneck depths."""

    def __init__(
        self,
        in_channels: int,
        base_planes: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
    ):
        """Initialize a 3D ResNet50V2-style encoder backbone.

        Args:
            in_channels: Input data channels.
            base_planes: Stage base widths (one per level).
            kernel_sizes: Per-stage kernel schedule used for compatibility metadata.
        """
        super().__init__()
        if len(base_planes) < 3:
            raise ValueError("base_planes must have at least 3 entries")

        p1 = base_planes[0]

        self.stem_conv = nn.Conv3d(in_channels, p1, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_norm = nn.InstanceNorm3d(p1, affine=True)
        self.stem_act = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Canonical ResNet50V2 depths for first four stages, then repeat 3 blocks.
        block_schedule = [3, 4, 6, 3] + [3] * max(0, len(base_planes) - 4)
        stages: list[nn.Module] = []
        for i, planes in enumerate(base_planes):
            if i == 0:
                stage_in = p1
                stride = 1
            else:
                stage_in = base_planes[i - 1] * BottleneckV2_3d.expansion
                stride = 2
            stages.append(ResNetV2Stage3d(stage_in, planes, num_blocks=block_schedule[i], stride=stride))
        self.stages = nn.ModuleList(stages)

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

    def forward(self, x: torch.Tensor, use_checkpoint: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode an input volume and return stem + multiscale feature maps.

        Args:
            x: Input tensor of shape ``(B, C, D, H, W)``.
            use_checkpoint: Whether to apply gradient checkpointing in residual stages.

        Returns:
            Tuple of (stem, stage_features), where stage_features is ordered
            shallow-to-deep.
        """
        ckpt = use_checkpoint and torch.is_grad_enabled()

        stem = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.stem_pool(stem)

        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = cast(torch.Tensor, _grad_ckpt(stage, x, use_reentrant=False)) if ckpt else stage(x)
            features.append(x)
        return stem, features


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
        unet_levels: int = 4,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        use_mamba: bool = False,
        use_checkpoint: bool = True,
        deep_reconstruction_head: bool = False,
    ):
        """Initialize the seismic reconstruction network.

        Args:
            input_channels: Number of input/output seismic channels.
            hidden_dims: Encoder stage widths (must match unet_levels).
            kernel_sizes: Optional odd kernel schedule for refinement blocks.
            unet_levels: Number of encoder/decoder levels.
            spatial_size: Intended patch size; retained for compatibility.
            use_mamba: Compatibility flag; ignored in this pure ResNetV2 variant.
            use_checkpoint: Enable stage-level gradient checkpointing.
            deep_reconstruction_head: Use a 2-layer head instead of a single 1x1x1 conv.
        """
        super().__init__()
        if use_mamba:
            print("WARNING: use_mamba is ignored in this architecture; using pure 3D ResNetV2-UNet blocks.")
        if unet_levels < 3 or unet_levels > 6:
            raise ValueError("unet_levels must be between 3 and 6")
        if len(hidden_dims) != unet_levels:
            raise ValueError(
                "hidden_dims length must match unet_levels "
                f"(got len(hidden_dims)={len(hidden_dims)}, unet_levels={unet_levels})"
            )

        self._stage_kernels = _resolve_stage_kernel_sizes(hidden_dims, kernel_sizes)
        self.use_checkpoint = use_checkpoint
        self.unet_levels = int(unet_levels)

        self.encoder = ResNet50V2Encoder3d(input_channels, hidden_dims, self._stage_kernels)
        stem_ch = hidden_dims[0]

        enc_out_channels = [h * BottleneckV2_3d.expansion for h in hidden_dims]
        dec_blocks: list[DecoderUpBlock3d] = []

        # Decoder blocks from deepest stage toward stem.
        for i in range(self.unet_levels - 1, 0, -1):
            dec_blocks.append(
                DecoderUpBlock3d(
                    in_channels=enc_out_channels[i],
                    skip_channels=enc_out_channels[i - 1],
                    out_channels=enc_out_channels[i - 1],
                    kernel_size=self._stage_kernels[i - 1],
                )
            )
        dec_blocks.append(
            DecoderUpBlock3d(
                in_channels=enc_out_channels[0],
                skip_channels=stem_ch,
                out_channels=stem_ch,
                kernel_size=self._stage_kernels[0],
            )
        )
        self.dec_blocks = nn.ModuleList(dec_blocks)

        # Backward-compatible named decoder blocks (e.g., dec4..dec1 for 4-level default).
        for i, block in enumerate(self.dec_blocks):
            level_name = self.unet_levels - i
            setattr(self, f"dec{level_name}", block)

        # Compatibility surface for existing decoder kernel regression tests.
        self.decoder = SimpleNamespace(
            dec_blocks=[
                SimpleNamespace(conv1=SimpleNamespace(kernel_size=(self._stage_kernels[i],) * 3))
                for i in range(self.unet_levels - 2, -1, -1)
            ]
        )

        self.final_up_interp = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_up_conv = nn.Conv3d(stem_ch, stem_ch, kernel_size=1, bias=False)

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
        """Predict reconstructed seismic amplitudes at full input resolution."""
        stem, stage_feats = self.encoder(x, use_checkpoint=self.use_checkpoint)

        y = stage_feats[-1]
        skips = list(reversed(stage_feats[:-1])) + [stem]
        for block, skip in zip(self.dec_blocks, skips):
            y = block(y, skip)

        y = self.final_up_interp(y)
        y = self.final_up_conv(y)
        if y.shape[-3:] != x.shape[-3:]:
            y = nn.functional.interpolate(y, size=x.shape[-3:], mode="nearest")
        return self.head(y).float()

    def swap_to_segmentation_head(self, n_classes: int = 1, freeze_body: bool = True) -> None:
        """Replace the reconstruction head with a segmentation-style logits head.

        Args:
            n_classes: Number of segmentation output channels.
            freeze_body: If True, freeze encoder/decoder weights after swap.
        """
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
                + list(self.dec_blocks.parameters())
                + list(self.final_up_conv.parameters())
            ):
                p.requires_grad = False

    def unfreeze_body(self) -> None:
        """Unfreeze encoder/decoder parameters for full fine-tuning."""
        for p in (
            list(self.encoder.parameters())
            + list(self.dec_blocks.parameters())
            + list(self.final_up_conv.parameters())
        ):
            p.requires_grad = True

    @property
    def head_type(self) -> str:
        """Return active head mode: reconstruction or segmentation."""
        return self._head_type

    @property
    def mamba_available(self) -> bool:
        """Report whether optional Mamba acceleration is available."""
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
    """Factory for SeismicUNet3d with compatibility-friendly arguments.

    Args:
        use_mamba: Compatibility flag accepted by training CLI.
        use_checkpoint: Enable gradient checkpointing in the encoder.
        kernel_sizes: Optional odd kernel schedule for refinement blocks.
        deep_reconstruction_head: Enable a deeper regression output head.
        **kwargs: Forwarded to SeismicUNet3d.

    Returns:
        An initialized SeismicUNet3d instance.
    """
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
