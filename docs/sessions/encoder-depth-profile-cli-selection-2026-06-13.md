# Encoder Depth Profile CLI Selection - 2026-06-13

## Context and Goals
- Add selectable encoder depth schedules through CLI for U-Net levels 3 and 4.
- Keep baseline behavior available while exposing two deeper presets.
- Ensure explicit per-stage override remains possible for experiments.

## What Was Done
- Added encoder depth profile resolution in the model stack.
- Added new CLI options:
  - --encoder_depth_profile with values: baseline, deeper, deepest
  - --encoder_stage_blocks for explicit per-stage block counts
- Wired options through training config and model creation.
- Added validation for explicit stage-block list length and positivity.
- Added tests for parser defaults/acceptance and model schedule mapping.
- Renamed profile labels from earlier names to baseline/deeper/deepest.

## How Was It Done
- Implemented a resolver that maps profile name + unet_levels to stage block schedules.
- Preserved canonical baseline schedule for compatibility.
- Added hard validation and descriptive errors for unsupported profile/level combinations.
- Updated training logs to print selected profile and resolved block schedule.
- Added regression tests covering:
  - unet_levels=4 baseline/deeper/deepest
  - unet_levels=3 baseline/deeper
  - unsupported combinations
  - explicit override precedence

## When Was It Done and By Whom
- Date: 2026-06-13
- Authoring agent: GitHub Copilot (GPT-5.3-Codex)
- Requested by: Donald PG

## Basic Info (Relevant Commits, Files Involved)
- Relevant commit(s): created in this session after summary generation.
- Files modified:
  - src/synthoseis_pre_train/models.py
  - src/synthoseis_pre_train/pretrain.py
  - train_cli.py
  - tests/test_model_kernel_defaults.py
  - tests/test_train_cli.py
- Files created:
  - docs/sessions/encoder-depth-profile-cli-selection-2026-06-13.md
  - docs/sessions/encoder-depth-profile-cli-selection-2026-06-13.html

## Next and/or Future Follow-Up Work Suggestions
- Add a quick benchmark script to compare epoch time for baseline vs deeper vs deepest.
- Optionally expose named schedules in experiment shell scripts for repeatability.
- Install pytest in the active environment and run targeted tests for full verification.
