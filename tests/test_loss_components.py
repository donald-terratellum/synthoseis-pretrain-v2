import torch

from synthoseis_pre_train.losses import LPIPSLoss, MultiComponentLoss3D, compute_pmse_loss


def test_compute_pmse_loss_finite_and_positive():
    recon = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16) * 5.0

    pmse = compute_pmse_loss(recon, target)

    assert torch.isfinite(pmse)
    assert float(pmse) > 0.0


def test_compute_pmse_loss_zero_for_identical_inputs():
    x = torch.ones(2, 1, 8, 8, 8)
    pmse = compute_pmse_loss(x, x)
    assert float(pmse) == 0.0


def test_lpips_disabled_returns_zero_for_3d_inputs():
    lpips = LPIPSLoss(enabled=False)
    x = torch.randn(2, 1, 12, 12, 12)
    y = torch.randn(2, 1, 12, 12, 12)
    loss = lpips(x, y)
    assert float(loss) == 0.0


def test_multi_component_matches_weighted_mse_pmse_mae_without_lpips():
    recon = torch.randn(2, 1, 12, 12, 12)
    target = torch.randn(2, 1, 12, 12, 12)

    mse_w, pmse_w, mae_w, lpips_w = 0.2, 0.6, 0.2, 0.0
    crit = MultiComponentLoss3D(
        mse_weight=mse_w,
        pmse_weight=pmse_w,
        mae_weight=mae_w,
        lpips_weight=lpips_w,
    )

    expected = (
        mse_w * torch.nn.functional.mse_loss(recon, target)
        + pmse_w * compute_pmse_loss(recon, target)
        + mae_w * torch.nn.functional.l1_loss(recon, target)
    )
    actual = crit(recon, target)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
