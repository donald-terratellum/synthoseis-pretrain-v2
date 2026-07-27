#!/usr/bin/env bash
set -euo pipefail

# Dry-run only: prints rm commands for checkpoint subfolders that would be deleted.
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-"/Volumes/Crucial X9/pretrain_v2_checkpoints"}
cd "${CHECKPOINTS_ROOT}"

for d in */; do
  d="${d%/}"
  case "$d" in
    checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper|\
    checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256f|\
    checkpoints_mse_pmse_mae_lpips__0-0-99-1__depth_3|\
    checkpoints_mse_pmse_mae_lpips__0-0-99-1__depth_4__48_84_146_256__deeper_3masks|\
    checkpoints_sliding_stats_mse_pmse_mae_lpips__0-0-100-0__depth_3|\
    checkpoints_sliding_stats_mse_pmse_mae_lpips__0-100-0-0__depth_3|\
    checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper_3masks)
      ;;
    *)
      echo rm -rf -- "$d"
      ;;
  esac
done
