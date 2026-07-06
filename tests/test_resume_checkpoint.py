from pathlib import Path

import torch

from synthoseis_pre_train.pretrain import _resolve_resume_checkpoint_path


def test_resolve_resume_checkpoint_path_uses_existing_file(tmp_path: Path):
    resume_path = tmp_path / "checkpoint_final_model.pt"
    resume_path.write_bytes(b"stub")

    assert _resolve_resume_checkpoint_path(resume_path) == resume_path


def test_resolve_resume_checkpoint_path_falls_back_to_latest_epoch_checkpoint(tmp_path: Path, capsys):
    missing_resume = tmp_path / "checkpoint_final_model.pt"
    older = tmp_path / "checkpoint_epoch_0003.pt"
    newer = tmp_path / "checkpoint_epoch_0007.pt"

    torch.save({"epoch": 3}, older)
    torch.save({"epoch": 7}, newer)

    resolved = _resolve_resume_checkpoint_path(missing_resume)
    out = capsys.readouterr().out

    assert resolved == newer
    assert "falling back to latest epoch checkpoint" in out


def test_resolve_resume_checkpoint_path_raises_when_no_candidates_exist(tmp_path: Path):
    missing_resume = tmp_path / "checkpoint_final_model.pt"

    try:
        _resolve_resume_checkpoint_path(missing_resume)
    except FileNotFoundError as exc:
        assert "Resume checkpoint not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
