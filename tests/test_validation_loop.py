from types import SimpleNamespace

from synthoseis_pre_train._validation_loop import _prepare_validation_dataset


class _LoaderWithLen:
    def __init__(self, dataset, n_batches=5):
        self.dataset = dataset
        self._n_batches = n_batches

    def __len__(self):
        return self._n_batches


def test_prepare_validation_dataset_uses_available_cubes(capsys):
    dataset = SimpleNamespace(available_cubes=["cube_a", "cube_b"])
    loader = _LoaderWithLen(dataset, n_batches=7)

    n = _prepare_validation_dataset(
        loader=loader,
        target_for_loader=None,
        ds_name="synthetic_ds",
        ds_idx=0,
        total_datasets=2,
    )

    assert n == 7
    out = capsys.readouterr().out
    assert "Val dataset synthetic_ds, cube_a, cube_b [1/2]" in out


def test_prepare_validation_dataset_falls_back_without_available_cubes(capsys):
    dataset = SimpleNamespace(data_path="/tmp/real/CostaRica_part1.npy")
    loader = _LoaderWithLen(dataset, n_batches=3)

    n = _prepare_validation_dataset(
        loader=loader,
        target_for_loader=None,
        ds_name="CostaRica_part1",
        ds_idx=1,
        total_datasets=4,
    )

    assert n == 3
    out = capsys.readouterr().out
    assert "Val dataset CostaRica_part1, CostaRica_part1 [2/4]" in out
