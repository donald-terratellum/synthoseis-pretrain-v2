from pathlib import Path

from synthoseis_pre_train.pretrain import _backup_dataset_folders


def test_backup_dataset_folders_copies_full_tree_and_preserves_file_times(tmp_path: Path):
    data_root = tmp_path / "data"
    backup_root = tmp_path / "backup"

    src_dataset_dir = data_root / "seismic__2026.70000000__synthoseis_run_1000"
    src_zarr_dir = src_dataset_dir / "model_data.zarr"
    src_zarr_dir.mkdir(parents=True)

    payload = src_zarr_dir / "payload.bin"
    payload.write_bytes(b"abc123")
    nested = src_zarr_dir / "nested" / "inner.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("hello", encoding="utf-8")

    original_mtime_ns = payload.stat().st_mtime_ns

    copied = _backup_dataset_folders(
        [str(src_zarr_dir)],
        data_root=data_root,
        backup_root=backup_root,
    )

    assert copied == [str(src_zarr_dir)]

    dst_dataset_dir = backup_root / "seismic__2026.70000000__synthoseis_run_1000"
    dst_payload = dst_dataset_dir / "model_data.zarr" / "payload.bin"
    dst_nested = dst_dataset_dir / "model_data.zarr" / "nested" / "inner.txt"

    assert dst_payload.exists()
    assert dst_payload.read_bytes() == b"abc123"
    assert dst_nested.exists()
    assert dst_nested.read_text(encoding="utf-8") == "hello"
    assert dst_payload.stat().st_mtime_ns == original_mtime_ns
