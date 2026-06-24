import csv
import random
from types import SimpleNamespace

from studies.run_random_training_sweep import (
    _load_used_run_config_fingerprints,
    _run_config_fingerprint,
    sample_run_config,
    sample_unique_run_config,
)



def test_sample_unique_run_config_skips_configs_already_in_csv(tmp_path):
    args = SimpleNamespace(encoder_depth_profile="deeper", encoder_stage_blocks=None)
    rng = random.Random(20260618)
    first_cfg = sample_run_config(rng, 1, tmp_path)
    first_fingerprint = _run_config_fingerprint(first_cfg, args)

    csv_path = tmp_path / "epoch_component_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unet_levels",
                "hidden_dims (as a space-delimited string)",
                "kernel_schedule (as a space-delimited string)",
                "encoder_depth_profile (profile - stage blocks)",
                "mse_weight",
                "pmse_weight",
                "mae_weight",
                "lpips_weight",
                "tv_weight",
                "gdl_weight",
            ]
        )
        writer.writerow(first_fingerprint)

    used_fingerprints = _load_used_run_config_fingerprints(csv_path)
    assert first_fingerprint in used_fingerprints

    rng = random.Random(20260618)
    unique_cfg = sample_unique_run_config(rng, 1, tmp_path, args, used_fingerprints)
    unique_fingerprint = _run_config_fingerprint(unique_cfg, args)

    assert unique_fingerprint != first_fingerprint
    assert unique_fingerprint not in {_run_config_fingerprint(first_cfg, args)}


def test_load_used_run_config_fingerprints_ignores_lr_column(tmp_path):
    csv_path = tmp_path / "epoch_component_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unet_levels",
                "hidden_dims (as a space-delimited string)",
                "kernel_schedule (as a space-delimited string)",
                "encoder_depth_profile (profile - stage blocks)",
                "lr",
                "mse_weight",
                "pmse_weight",
                "mae_weight",
                "lpips_weight",
                "tv_weight",
                "gdl_weight",
            ]
        )
        row_base = [
            "4",
            "32 64 128 256",
            "3 3 3 3",
            "deeper - 3 5 8 12",
        ]
        row_tail = ["0.000000", "0.000000", "0.990000", "0.010000", "0.000000", "0.000000"]
        writer.writerow(row_base + ["1.00000000e-05"] + row_tail)
        writer.writerow(row_base + ["5.00000000e-06"] + row_tail)

    fingerprints = _load_used_run_config_fingerprints(csv_path)
    assert len(fingerprints) == 1
