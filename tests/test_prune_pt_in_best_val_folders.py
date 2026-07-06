from pathlib import Path

from studies.prune_pt_in_best_val_folders import should_keep


def test_should_keep_preserves_final_resumable_checkpoint():
    assert should_keep(Path("checkpoint_final_model.pt")) is True


def test_should_keep_preserves_existing_best_and_final_artifacts():
    assert should_keep(Path("best_val_epoch.pt")) is True
    assert should_keep(Path("final_model.pt")) is True


def test_should_keep_prunes_non_kept_epoch_checkpoints():
    assert should_keep(Path("checkpoint_epoch_0001.pt")) is False
    assert should_keep(Path("checkpoint_epoch_0005.pt")) is True


def test_should_keep_uses_configurable_keep_every():
    assert should_keep(Path("checkpoint_epoch_0010.pt"), keep_every=10) is True
    assert should_keep(Path("checkpoint_epoch_0015.pt"), keep_every=10) is False
