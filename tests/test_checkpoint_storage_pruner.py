import csv

from studies.run_random_training_sweep import enforce_checkpoint_storage_budget



def _touch(path, size_bytes=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"x" * size_bytes)



def test_checkpoint_pruner_keeps_top5_and_prunes_others(tmp_path):
    checkpoints_root = tmp_path / "checkpoints"
    csv_path = checkpoints_root / "epoch_component_metrics.csv"

    run_names = [f"run_{i:02d}" for i in range(6)]
    for i, name in enumerate(run_names):
        run_dir = checkpoints_root / name
        runs_dir = run_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        _touch(run_dir / "final_model.pt", 16)
        _touch(run_dir / "checkpoint_epoch_0001.pt", 16)
        _touch(run_dir / "checkpoint_epoch_0002.pt", 16)
        _touch(run_dir / "checkpoint_epoch_0003.pt", 16)
        _touch(run_dir / "final_model_raw.pt", 16)
        _touch(runs_dir / "events.out.tfevents", 16)

    headers = [
        "tensorboard folder",
        "epoch",
        "validation mae/L1",
        "validation mse",
        "validation LPIPS",
    ]
    rows = []
    # run_00..run_04 are best 5, run_05 is worst and should be pruned.
    for i, name in enumerate(run_names):
        tb = f"checkpoints/{name}/runs"
        rows.append([tb, "1", f"{1.0 + i:.4f}", "0.0", "0.0"])
        rows.append([tb, "2", f"{0.5 + i:.4f}", "0.0", "0.0"])
        rows.append([tb, "3", f"{0.8 + i:.4f}", "0.0", "0.0"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    summary = enforce_checkpoint_storage_budget(
        csv_path=csv_path,
        checkpoints_root=checkpoints_root,
        max_bytes=10**9,
        top_k_keep_all=5,
    )

    assert summary["ranked_runs"] == 6
    assert summary["topk_kept_all"] == 5

    pruned_dir = checkpoints_root / "run_05"
    kept = sorted(p.name for p in pruned_dir.iterdir())
    assert kept == ["checkpoint_epoch_0002.pt", "final_model.pt"]

    top_dir = checkpoints_root / "run_00"
    top_names = sorted(p.name for p in top_dir.iterdir())
    assert "runs" in top_names
    assert "final_model_raw.pt" in top_names
    assert "checkpoint_epoch_0001.pt" in top_names
