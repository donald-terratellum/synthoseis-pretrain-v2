import numpy as np
from synthoseis_pre_train.masking import (
    create_mask_3d,
    apply_mask_to_seismic,
    keep_trace_extrema_only,
    apply_input_random_sparse_keep,
    apply_input_decimate_trilinear,
    apply_input_trace_dropout,
)


def test_create_mask_3d_all_masked_when_full_ratio_and_prob():
    # Create small seismic cube (z, x, y)
    z, x, y = 8, 6, 6
    seismic = np.random.RandomState(0).randn(z, x, y).astype(np.float32)

    # If trace_mask_ratio=1.0 and cluster_prob=1.0, every trace index is chosen and
    # each masks a 3x3 region; the net effect should be that all traces are masked
    mask = create_mask_3d(seismic, trace_mask_ratio=1.0, cluster_prob=1.0, random_seed=42)

    # Entire mask should be False (nothing preserved)
    assert mask.shape == seismic.shape
    assert not mask.any(), "Expected all entries to be masked when ratio=1.0 and prob=1.0"


def test_peaks_troughs_preserved_when_no_trace_masking():
    # Construct seismic with a clear peak at z=3 for a single trace (x=2,y=1)
    z, x, y = 7, 5, 4
    seismic = np.zeros((z, x, y), dtype=np.float32)

    tx, ty = 2, 1
    # Create a local peak at z=3
    seismic[2, tx, ty] = 0.1
    seismic[3, tx, ty] = 1.0  # peak
    seismic[4, tx, ty] = 0.05

    # No random trace masking: trace_mask_ratio=0.0
    mask = create_mask_3d(seismic, trace_mask_ratio=0.0, cluster_prob=0.0, random_seed=7)

    # Only the local extrema (z=3) for that trace should be True; neighbors along z should be False
    assert mask.shape == seismic.shape
    assert bool(mask[3, tx, ty])
    # neighbors should be masked (False)
    assert not bool(mask[2, tx, ty])
    assert not bool(mask[4, tx, ty])


def test_apply_mask_to_seismic_fills_with_zero():
    z, x, y = 6, 4, 4
    seismic = np.arange(z * x * y, dtype=np.float32).reshape((z, x, y))
    # Create a mask that preserves only one voxel
    mask = np.zeros_like(seismic, dtype=bool)
    mask[1, 1, 1] = True

    masked_data, original, used_mask = apply_mask_to_seismic(seismic, mask, fill_value=0.0)

    # masked positions should be zero
    assert masked_data[0, 0, 0] == 0.0
    # preserved voxel should remain original
    assert masked_data[1, 1, 1] == original[1, 1, 1]
    # returned mask should be identical
    assert np.array_equal(used_mask, mask)


def test_keep_trace_extrema_only_retains_local_extrema_values():
    z, x, y = 7, 3, 2
    seismic = np.zeros((z, x, y), dtype=np.float32)
    seismic[:, 1, 1] = np.array([0.0, 0.1, 1.0, 0.2, 2.0, 0.1, 0.0], dtype=np.float32)

    out = keep_trace_extrema_only(seismic)

    assert out.shape == seismic.shape
    # Interior local extrema at z=2 (peak), z=3 (trough), z=4 (peak) are kept.
    assert out[2, 1, 1] == seismic[2, 1, 1]
    assert out[3, 1, 1] == seismic[3, 1, 1]
    assert out[4, 1, 1] == seismic[4, 1, 1]
    # Non-extrema neighbors are zeroed.
    assert out[1, 1, 1] == 0.0
    assert out[5, 1, 1] == 0.0


def test_apply_input_random_sparse_keep_preserves_some_voxels():
    seismic = np.random.RandomState(1).randn(8, 8, 8).astype(np.float32)
    out = apply_input_random_sparse_keep(
        seismic,
        fraction_min=0.2,
        fraction_max=0.2,
        method="uniform",
    )

    kept = int(np.count_nonzero(out))
    assert out.shape == seismic.shape
    assert kept > 0
    assert kept < seismic.size


def test_apply_input_decimate_trilinear_preserves_anchor_samples():
    seismic = np.random.RandomState(2).randn(9, 9, 9).astype(np.float32)
    out = apply_input_decimate_trilinear(seismic, parity=0)

    assert out.shape == seismic.shape
    assert np.allclose(out[0::2, 0::2, 0::2], seismic[0::2, 0::2, 0::2])


def test_apply_input_trace_dropout_masks_full_traces():
    seismic = np.ones((8, 8, 8), dtype=np.float32)
    out = apply_input_trace_dropout(seismic, trace_mask_ratio=1.0, cluster_prob=1.0, random_seed=3)

    assert out.shape == seismic.shape
    assert np.count_nonzero(out) == 0
