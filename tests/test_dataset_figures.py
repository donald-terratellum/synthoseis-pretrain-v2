import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from synthoseis_pre_train._dataset_figures import _log_per_dataset_figures


class _SourceTaggedTinyDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        shape = (8, 8, 8)
        inp = torch.ones(shape, dtype=torch.float32)
        tgt = torch.ones(shape, dtype=torch.float32)
        mask = torch.ones(shape, dtype=torch.bool)
        return inp, tgt, mask, "R"


class _RecordingWriter:
    def __init__(self):
        self.tags = []

    def add_figure(self, tag, fig, global_step):
        self.tags.append((tag, global_step))


def test_log_per_dataset_figures_accepts_source_tagged_samples(monkeypatch):
    calls = {"count": 0}

    def _fake_make_4panel_figure(*args, **kwargs):
        calls["count"] += 1
        return plt.figure()

    monkeypatch.setattr("synthoseis_pre_train._dataset_figures.make_4panel_figure", _fake_make_4panel_figure)

    ds_a = _SourceTaggedTinyDataset("/tmp/ds_a/model_data.zarr")
    ds_b = _SourceTaggedTinyDataset("/tmp/ds_b/model_data.zarr")
    loader = DataLoader(ConcatDataset([ds_a, ds_b]), batch_size=1, shuffle=False)

    model = nn.Identity()
    writer = _RecordingWriter()

    _log_per_dataset_figures(
        model=model,
        merged_loader=loader,
        device=torch.device("cpu"),
        writer=writer,
        epoch=0,
        epoch_loss=0.123,
    )

    assert calls["count"] == 2
    assert len(writer.tags) == 2
    assert writer.tags[0][0] == "train/ds_a"
    assert writer.tags[1][0] == "train/ds_b"
