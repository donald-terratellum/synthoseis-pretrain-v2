import re

import torch

from synthoseis_pre_train.models import report_masked_voxel_stats



def _extract_percentages(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)%", text)]



def test_report_masked_voxel_stats_stays_bounded_for_stagewise_counts(capsys):
    target = torch.ones(1, 1, 4, 4, 4)
    input_data = target.clone()

    # Keep 75% of traces and 50% of voxels within the surviving traces.
    input_data[:, :, :, :, 1:] = 0.0
    input_data[:, :, :, 0, 0] = 0.0

    report_masked_voxel_stats(input_data, target=target)
    out = capsys.readouterr().out
    percentages = _extract_percentages(out)

    assert percentages, out
    assert all(0.0 <= value <= 100.0 for value in percentages), out


def test_report_masked_voxel_stats_total_is_bounded_and_uses_arrow_format(capsys):
    target = torch.ones(1, 1, 4, 4, 4)
    input_data = target.clone()

    report_masked_voxel_stats(input_data, target=target)
    out = capsys.readouterr().out

    assert "-->" in out, out

    match = re.search(r"-->\s*([0-9]+(?:\.[0-9]+)?)%", out)
    assert match is not None, out
    total_pct = float(match.group(1))
    assert 0.0 <= total_pct <= 100.0, out
