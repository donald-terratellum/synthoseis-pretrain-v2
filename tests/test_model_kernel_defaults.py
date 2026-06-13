"""Regression checks for kernel-size scheduling defaults."""

import pytest
import torch

from synthoseis_pre_train.models import ResBlock3d, create_model


def test_default_model_keeps_legacy_3x3_residual_kernels():
    """Default constructor should match pre-change 3x3 residual kernels."""
    model = create_model(use_checkpoint=False)

    res_blocks = [m for m in model.modules() if isinstance(m, ResBlock3d)]
    assert res_blocks, "Expected ResBlock3d modules in the default model"

    for block in res_blocks:
        assert block.conv1.kernel_size == (3, 3, 3)
        assert block.conv2.kernel_size == (3, 3, 3)


def test_custom_schedule_changes_shallow_kernels_but_not_bottleneck():
    """Custom schedule should apply per-stage kernels while bottleneck remains 3."""
    model = create_model(
        hidden_dims=(16, 32, 64, 128),
        kernel_sizes=(7, 5, 3, 3),
        use_checkpoint=False,
    )

    assert model.encoder.stem.conv1.kernel_size == (7, 7, 7)
    assert model.encoder.stem.conv2.kernel_size == (7, 7, 7)

    assert model.encoder.enc_blocks[0].conv1.kernel_size == (5, 5, 5)
    assert model.encoder.enc_blocks[1].conv1.kernel_size == (3, 3, 3)
    assert model.encoder.enc_blocks[2].conv1.kernel_size == (3, 3, 3)

    assert model.encoder.bottleneck.conv1.kernel_size == (3, 3, 3)
    assert model.encoder.bottleneck.conv2.kernel_size == (3, 3, 3)

    # Decoder mirrors encoder scales from deep->shallow: (3, 5, 7)
    assert model.decoder.dec_blocks[0].conv1.kernel_size == (3, 3, 3)
    assert model.decoder.dec_blocks[1].conv1.kernel_size == (5, 5, 5)
    assert model.decoder.dec_blocks[2].conv1.kernel_size == (7, 7, 7)


def test_model_initializes_for_unet_levels_3_to_6():
    configs = [
        (3, (16, 32, 64)),
        (4, (16, 32, 64, 128)),
        (5, (8, 16, 32, 64, 128)),
        (6, (8, 16, 32, 64, 128, 256)),
    ]

    for levels, hidden_dims in configs:
        model = create_model(use_checkpoint=False, unet_levels=levels, hidden_dims=hidden_dims)
        assert model.unet_levels == levels
        assert len(model.dec_blocks) == levels


def test_model_forward_for_unet_levels_3_to_5():
    # Keep this forward test lightweight; level 6 requires a very large spatial
    # input to keep bottleneck normalization valid.
    configs = [
        (3, (16, 32, 64), (64, 64, 64)),
        (4, (16, 32, 64, 128), (64, 64, 64)),
        (5, (8, 16, 32, 64, 128), (128, 128, 128)),
    ]

    for levels, hidden_dims, shape in configs:
        model = create_model(use_checkpoint=False, unet_levels=levels, hidden_dims=hidden_dims)
        model.eval()
        x = torch.randn(1, 1, *shape)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 1, *shape)


def test_encoder_depth_profile_schedules_for_unet_levels_4():
    baseline = create_model(use_checkpoint=False, unet_levels=4, hidden_dims=(16, 32, 64, 128))
    cheap = create_model(
        use_checkpoint=False,
        unet_levels=4,
        hidden_dims=(16, 32, 64, 128),
        encoder_depth_profile="deeper",
    )
    moderate = create_model(
        use_checkpoint=False,
        unet_levels=4,
        hidden_dims=(16, 32, 64, 128),
        encoder_depth_profile="deepest",
    )

    assert baseline.encoder.stage_block_schedule == (3, 4, 6, 3)
    assert cheap.encoder.stage_block_schedule == (3, 4, 8, 4)
    assert moderate.encoder.stage_block_schedule == (3, 5, 8, 5)


def test_encoder_depth_profile_schedules_for_unet_levels_3():
    baseline = create_model(use_checkpoint=False, unet_levels=3, hidden_dims=(16, 32, 64))
    cheap = create_model(
        use_checkpoint=False,
        unet_levels=3,
        hidden_dims=(16, 32, 64),
        encoder_depth_profile="deeper",
    )

    assert baseline.encoder.stage_block_schedule == (3, 4, 6)
    assert cheap.encoder.stage_block_schedule == (3, 5, 8)


def test_encoder_depth_profile_rejects_unsupported_profile_for_levels():
    with pytest.raises(ValueError, match="encoder_depth_profile"):
        create_model(
            use_checkpoint=False,
            unet_levels=3,
            hidden_dims=(16, 32, 64),
            encoder_depth_profile="deepest",
        )


def test_explicit_encoder_stage_blocks_override_profile():
    model = create_model(
        use_checkpoint=False,
        unet_levels=4,
        hidden_dims=(16, 32, 64, 128),
        encoder_depth_profile="deepest",
        encoder_stage_blocks=(2, 3, 4, 5),
    )
    assert model.encoder.stage_block_schedule == (2, 3, 4, 5)
