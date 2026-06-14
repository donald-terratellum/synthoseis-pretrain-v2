# Session Summary: Masking, Dataloader, Loss Diagnostics, and GDL

**Date:** 2026-06-14  
**Branch:** `feature/multi-loss-and-unet-levels`

---

## Context and Goals

This session focused on improving the seismic pre-training pipeline in four connected areas:

1. Plan and implement new input masking strategies and ensure clustered full-trace masking is always applied after the selected per-sample masking strategy.
2. Investigate training slowdowns and silent failures, especially around masking and dataloader behavior.
3. Improve perceptual and structural training losses, including LPIPS behavior and a new Gradient Difference Loss (GDL) term for suppressing stripe artifacts caused by zeroed trace clusters.
4. Add targeted diagnostics, real-data masking reports, and CSV reporting updates so training behavior is easier to interpret and compare.

---

## What Was Done

### 1. Input masking feature implementation

- Added three configurable per-sample input masking strategies:
  - extrema-only retention
  - sparse voxel keep
  - decimate + trilinear reconstruction
- Enforced post-strategy clustered trace dropout so x-only zeroed traces are always applied after whichever masking strategy was chosen.
- Added CLI/config plumbing for the new masking probabilities.

### 2. LPIPS improvement

- Verified and updated LPIPS for 3D volumes so it uses three orthogonal middle planes rather than only two vertical slices.

### 3. CLI and wrapper updates

- Added underscore and hyphen forms for the new masking flags in the training CLI.
- Updated the multi-dataset shell wrapper to expose the same masking controls.

### 4. Runtime investigation and diagnostics

- Investigated silent training failure symptoms and narrowed likely cause to external termination / resource pressure rather than an in-process Python traceback.
- Added removable timing diagnostics to masking functions and to train-batch fetch timing.
- Built a real-data masking report test using a synthoseis-generated Zarr store, printing retained / changed / masked percentages and elapsed time for each masking method.

### 5. Dataloader simplification

- Refactored the non-cached dataloader path to stop materializing full cubes per sample.
- It now passes zarr array handles into extraction helpers so only the selected subvolume window is read.

### 6. Poisson masking performance repair

- Optimized the Poisson-like sparse keep path to avoid multi-minute runtimes on dense keep fractions.
- Added an adaptive high-density fallback and a cheaper low-density rejection path.

### 7. New GDL loss support

- Implemented `mc_gdl_weight` in `MultiComponentLoss3D`.
- Added Gradient Difference Loss (GDL) to suppress stripe artifacts while preserving true geological discontinuities.
- Added `gdl_weight` plus `train gdl` / `validation gdl` reporting in the per-epoch component metrics CSV.

---

## How It Was Done

- Used focused code reads around the controlling paths in:
  - masking
  - dataloader
  - augmentation
  - pretrain runtime
  - loss construction and reporting
- Implemented changes incrementally using small patch edits.
- Validated after each substantive change with targeted pytest runs and small benchmarks.
- Compared train/validation PMSE behavior against code paths and git history to confirm there was no hidden validation-only bug in the PMSE implementation.

Key validation actions completed during the session:

- `uv run pytest tests/test_masking_behavior.py`
- `uv run pytest tests/test_loss_components.py`
- `uv run pytest tests/test_train_cli.py`
- `uv run pytest tests/test_merged_dataloader.py`
- `uv run pytest tests/test_masking_methods_report.py -s -v`
- Targeted Poisson benchmark on a `128^3` volume at `20%` keep fraction

---

## When and By Whom

- **Date:** 2026-06-14
- **Primary author:** GitHub Copilot
- **Directed by:** Donald P. Griffith
- **Repository branch:** `feature/multi-loss-and-unet-levels`

---

## Basic Info

### Relevant Files Involved

- `README.md`
- `src/synthoseis_pre_train/_criterion.py`
- `src/synthoseis_pre_train/augmentation.py`
- `src/synthoseis_pre_train/dataloader.py`
- `src/synthoseis_pre_train/losses.py`
- `src/synthoseis_pre_train/masking.py`
- `src/synthoseis_pre_train/pretrain.py`
- `tests/test_loss_components.py`
- `tests/test_masking_behavior.py`
- `tests/test_masking_methods_report.py`
- `tests/test_merged_dataloader.py`
- `tests/test_train_cli.py`
- `train_cli.py`
- `train_multi_datasets.sh`

### Relevant Commit Context

- This summary captures working-tree changes made during the 2026-06-14 session.
- The resulting session commit and push details are reported after commit/push completion.

---

## Next / Future Follow-Up Work Suggestions

- Add a targeted test that exercises `mc_gdl_weight` numerically and verifies the new CSV columns explicitly.
- Consider a lateral-only TV variant (`x/y` only) so TV can be compared more directly against GDL for stripe suppression.
- Add optional train/validation timing diagnostics around forward, loss, backward, and optimizer step to complement the fetch timing already added.
- Run a controlled ablation across:
  - MAE + LPIPS
  - MAE + LPIPS + TV
  - MAE + LPIPS + GDL
  - MAE + LPIPS + TV + GDL
- Revisit masked-loss support after artifact behavior stabilizes, since the current loss still supervises the full target volume.