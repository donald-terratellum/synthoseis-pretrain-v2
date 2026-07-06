# Top-10 Retrain Loop, Prune, and Resume Fixes — 2026-07-06

## context and goals
- Make unattended training of the top-10 model configurations restartable across multi-day runs.
- Avoid manual checkpoint copying and support automatic resume from `checkpoint_final_model.pt` when present.
- Ensure pruning keeps the resumable checkpoint artifacts needed for restart.
- Add explicit cleanup of previously generated synthetic datasets before creating the next batch.
- Add a small shell wrapper so the long manual experiment block can be replaced by one command.

## what was done
- Added `studies/run_top10_retrain_loop.py` to orchestrate repeated data generation, training, and pruning for the top-10 configurations.
- Added resume/state tracking so a stopped run can continue from its saved loop position.
- Added automatic resume selection in training: use `checkpoint_final_model.pt` when present, otherwise fall back to the latest epoch checkpoint.
- Added explicit deletion of previous generated synthetic dataset folders before generating the next pass.
- Added a shell wrapper function in `train_cli__experiments.sh` to call the new loop CLI with preferred defaults.
- Extended prune behavior so `checkpoint_final_model.pt` is preserved.
- Added `--keep-every` to the prune CLI so epoch checkpoint retention can be changed from the default of 5 to other values such as 10.
- Added regression tests for prune retention, resume fallback, top-10 loop behavior, and dataset cleanup.
- Saved a resumable overwriteable final checkpoint at `checkpoint_final_model.pt` on the last training epoch.

## how was it done
- Implemented a loop driver that reads and writes a JSON state file, then restores pass/model/start-index position on restart.
- Encoded the 10 model definitions as structured specs inside the loop script so the same architectures/loss weights can be retrained unattended.
- Added a helper to remove prior generated dataset folders by run-index range before each new data-generation pass.
- Modified `train_cli.py` / `pretrain.py` integration so the loop can pass cumulative epoch targets and automatically resume when a final checkpoint exists.
- Updated `studies/prune_pt_in_best_val_folders.py` to accept a configurable checkpoint retention increment and to keep the new final resumable checkpoint.
- Added tests that exercise the new pruning increment, deletion helper, and resume fallback behavior.

## when was it done and by whom
- Date/time: 2026-07-06 15:34:10 CDT.
- Authoring agent: GitHub Copilot (GPT-5.4 mini).

## basic info (relevant commits, files involved)
- Branch: `feature/multi-loss-and-unet-levels`.
- Commit created in this closeout: see pushed commit reported in terminal output.
- Files involved:
  - `studies/run_top10_retrain_loop.py`
  - `studies/prune_pt_in_best_val_folders.py`
  - `train_cli__experiments.sh`
  - `src/synthoseis_pre_train/pretrain.py`
  - `tests/test_top10_retrain_loop.py`
  - `tests/test_prune_pt_in_best_val_folders.py`
  - `tests/test_resume_checkpoint.py`
- Validation commands run during this session:
  - `uv run pytest tests/test_top10_retrain_loop.py tests/test_prune_pt_in_best_val_folders.py tests/test_resume_checkpoint.py -q`

## next and/or future follow-up work suggestions
- Add a single manifest file for the 10 model specs so the loop script does not need hardcoded configuration.
- Add an integration smoke test that exercises one full loop pass with dummy dataset generation and pruning.
- Consider adding a second resume priority before the fallback epoch checkpoint if you want best-val to be preferred over the latest epoch in some workflows.
- Add a README snippet documenting the one-line wrapper function and the restart/resume semantics.
