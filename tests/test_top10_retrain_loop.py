from pathlib import Path

from studies.run_top10_retrain_loop import (
    TOP_10_MODEL_SPECS,
    _build_train_command,
    _delete_generated_datasets,
    cumulative_targets,
    parse_schedule,
)


def test_parse_schedule_and_cumulative_targets():
    increments = parse_schedule("2,2,3,3,5,5,5,5")
    assert increments == [2, 2, 3, 3, 5, 5, 5, 5]
    assert cumulative_targets(increments) == [2, 4, 7, 10, 15, 20, 25, 30]


def test_build_train_command_omits_resume_when_checkpoint_missing(tmp_path: Path):
    spec = TOP_10_MODEL_SPECS[0]
    spec_copy = type(spec)(
        name=spec.name,
        output_dir=str(tmp_path / "model_a"),
        unet_levels=spec.unet_levels,
        hidden_dims=spec.hidden_dims,
        encoder_depth_profile=spec.encoder_depth_profile,
        mc_mae_weight=spec.mc_mae_weight,
        mc_lpips_weight=spec.mc_lpips_weight,
        mc_tv_weight=spec.mc_tv_weight,
        mc_gdl_weight=spec.mc_gdl_weight,
        lr=spec.lr,
    )

    cmd = _build_train_command(
        spec_copy,
        epoch_target=7,
        data_folder="/tmp/data",
        real_train_paths="/tmp/real_train",
        real_test_paths="/tmp/real_test",
        epoch_samples=200000,
        real_epoch_samples=50000,
        test_batches_per_epoch=60,
        train_batches_per_epoch=100,
        val_batches_per_epoch=40,
        batch_size=2,
        val_split_ratio=0.3,
        lr_min=5e-6,
    )

    assert "--resume" not in cmd


def test_build_train_command_includes_resume_when_checkpoint_exists(tmp_path: Path):
    spec = TOP_10_MODEL_SPECS[0]
    out_dir = tmp_path / "model_b"
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoint_final_model.pt").write_bytes(b"stub")

    spec_copy = type(spec)(
        name=spec.name,
        output_dir=str(out_dir),
        unet_levels=spec.unet_levels,
        hidden_dims=spec.hidden_dims,
        encoder_depth_profile=spec.encoder_depth_profile,
        mc_mae_weight=spec.mc_mae_weight,
        mc_lpips_weight=spec.mc_lpips_weight,
        mc_tv_weight=spec.mc_tv_weight,
        mc_gdl_weight=spec.mc_gdl_weight,
        lr=spec.lr,
    )

    cmd = _build_train_command(
        spec_copy,
        epoch_target=10,
        data_folder="/tmp/data",
        real_train_paths="/tmp/real_train",
        real_test_paths="/tmp/real_test",
        epoch_samples=200000,
        real_epoch_samples=50000,
        test_batches_per_epoch=60,
        train_batches_per_epoch=100,
        val_batches_per_epoch=40,
        batch_size=2,
        val_split_ratio=0.3,
        lr_min=5e-6,
    )

    assert "--resume" in cmd
    idx = cmd.index("--resume")
    assert cmd[idx + 1].endswith("checkpoint_final_model.pt")


def test_delete_generated_datasets_deletes_matching_run_indices(tmp_path: Path):
    keep_dir = tmp_path / "seismic__2026.50000000__synthoseis_run_1002"
    delete_a = tmp_path / "seismic__2026.50000000__synthoseis_run_1000"
    delete_b = tmp_path / "seismic__2026.50000000__synthoseis_run_1001"
    keep_dir.mkdir()
    delete_a.mkdir()
    delete_b.mkdir()

    deleted = _delete_generated_datasets(
        data_folder=str(tmp_path),
        start_index=1000,
        count=2,
        dry_run=False,
    )

    assert deleted == 2
    assert not delete_a.exists()
    assert not delete_b.exists()
    assert keep_dir.exists()


def test_delete_generated_datasets_dry_run_preserves_files(tmp_path: Path):
    ds_dir = tmp_path / "seismic__2026.50000000__synthoseis_run_1000"
    ds_dir.mkdir()

    deleted = _delete_generated_datasets(
        data_folder=str(tmp_path),
        start_index=1000,
        count=1,
        dry_run=True,
    )

    assert deleted == 1
    assert ds_dir.exists()
