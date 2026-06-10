"""Regression tests for resize-convolution decoder upsampling."""

import torch
import torch.nn as nn

from synthoseis_pre_train.models import DecoderUpBlock3d, create_model


def test_decoder_block_uses_resize_convolution_layers():
    block = DecoderUpBlock3d(in_channels=64, skip_channels=32, out_channels=32, kernel_size=3)

    assert isinstance(block.up_interp, nn.Upsample)
    assert block.up_interp.scale_factor == 2.0
    assert block.up_interp.mode == "nearest"
    assert isinstance(block.up_conv, nn.Conv3d)
    assert block.up_conv.kernel_size == (1, 1, 1)


def test_decoder_block_matches_skip_shape_after_alignment():
    torch.manual_seed(0)
    block = DecoderUpBlock3d(in_channels=64, skip_channels=32, out_channels=32, kernel_size=3)

    x = torch.randn(2, 64, 8, 8, 8)
    skip = torch.randn(2, 32, 15, 16, 17)

    y = block(x, skip)
    assert y.shape == (2, 32, 15, 16, 17)


def test_model_forward_shape_and_gradients_work_with_resize_decoder():
    torch.manual_seed(0)
    model = create_model(use_checkpoint=False, hidden_dims=(8, 16, 32, 64))

    # Ensure transpose-conv upsamplers are no longer present in decoder/final-up path.
    assert not any(isinstance(m, nn.ConvTranspose3d) for m in model.modules())

    x = torch.randn(1, 1, 64, 64, 64, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape

    loss = (y - x).pow(2).mean()
    loss.backward()

    assert model.dec4.up_conv.weight.grad is not None
    assert model.dec3.up_conv.weight.grad is not None
    assert model.dec2.up_conv.weight.grad is not None
    assert model.dec1.up_conv.weight.grad is not None
    assert model.final_up_conv.weight.grad is not None
