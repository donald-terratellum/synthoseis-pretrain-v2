from pathlib import Path

import numpy as np
import zarr

from synthoseis_pre_train.vae_export import (
    default_output_zarr_path,
    default_radius_from_subset_size,
    export_vae_dataset,
    infer_dataset_id_from_path,
    main,
)


def _create_src_model_data_zarr(path: Path, shape=(20, 20, 20)) -> Path:
    root = zarr.open(str(path), mode="w")

    seismic = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    geoscore = np.ones(shape, dtype=np.float32)

    if hasattr(root, "create_array"):
        root.create_array("seismicCubes_cumsum_fullstack", data=seismic, chunks=(10, 10, 10))
        root.create_array("geologic_score", data=geoscore, chunks=(10, 10, 10))
    else:
        root.create_dataset(
            "seismicCubes_cumsum_fullstack",
            shape=seismic.shape,
            data=seismic,
            chunks=(10, 10, 10),
        )
        root.create_dataset(
            "geologic_score",
            shape=geoscore.shape,
            data=geoscore,
            chunks=(10, 10, 10),
        )

    return path


def test_default_radius_from_subset_size_matches_contract():
    radius = default_radius_from_subset_size((32, 32, 64))
    assert radius == (32.0 / 3.0)


def test_default_output_name_uses_train_4digit(tmp_path: Path):
    out = default_output_zarr_path(dataset_id=259, output_root=tmp_path)
    assert out == tmp_path / "train_0259.zarr"


def test_export_writes_expected_patch_tensor_shape_and_axis_order(tmp_path: Path):
    src = _create_src_model_data_zarr(tmp_path / "model_data.zarr", shape=(20, 20, 20))
    out = tmp_path / "train_0001.zarr"

    written = export_vae_dataset(
        dataset_zarr_path=src,
        dataset_id=1,
        subset_size_xyz=(4, 5, 6),
        n_subsets=12,
        output_zarr_path=out,
        score_min=0.0,
        candidate_count=80,
        candidate_probes=8,
        seed=123,
    )

    assert written == out
    dst = zarr.open(str(out), mode="r")
    patches = np.asarray(dst["patches"])

    assert patches.shape == (12, 4, 5, 6)
    assert patches.dtype == np.float32
    assert int(dst.attrs["dataset_id"]) == 1
    assert list(dst.attrs["subset_size_xyz"]) == [4, 5, 6]


def test_infer_dataset_id_from_path_reads_synthoseis_run_suffix():
    inferred = infer_dataset_id_from_path(
        Path("/tmp/seismic__20260101_synthoseis_run_0259/model_data.zarr")
    )
    assert inferred == 259


def test_cli_main_writes_default_train_name_under_output_root(tmp_path: Path):
    src = _create_src_model_data_zarr(
        tmp_path / "seismic__20260604_synthoseis_run_0007" / "model_data.zarr",
        shape=(24, 24, 24),
    )

    ret = main(
        [
            "--dataset-zarr",
            str(src),
            "--n-subsets",
            "6",
            "--subset-size-x",
            "4",
            "--subset-size-y",
            "4",
            "--subset-size-z",
            "8",
            "--candidate-count",
            "64",
            "--candidate-probes",
            "8",
            "--score-min",
            "0.0",
            "--vae-output-root",
            str(tmp_path / "vae_data"),
        ]
    )

    assert ret == 0
    out = tmp_path / "vae_data" / "train_0007.zarr"
    assert out.exists()

    dst = zarr.open(str(out), mode="r")
    assert tuple(dst["patches"].shape) == (6, 4, 4, 8)
