#####################################################

### mae only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 1.0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 4 --hidden_dims 32 64 128 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_4.log

### mae only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 1.0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256f \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256f.log

uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 1.0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper.log

uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .95 \
  --mc_lpips_weight .05 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_84_146_256__deeper_3masks.log

uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .99 \
  --mc_lpips_weight .01 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-99-1__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-99-1__depth_4__48_84_146_256__deeper_3masks.log

uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .01 \
  --mc_lpips_weight .99 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-1-99__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-1-99__depth_4__48_84_146_256__deeper_3masks.log


uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
    --mc_lpips_calib_weight 0.05 \
    --mc_mae_weight 0.05 \
    --mc_lpips_weight 0.90 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-5-95__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-5-95__depth_4__48_84_146_256__deeper_3masks.log


uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
    --mc_lpips_calib_weight 0.03 \
    --mc_mae_weight 0.05 \
    --mc_lpips_weight 0.92 \
    --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-5-92_2__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-5-92_2__depth_4__48_84_146_256__deeper_3masks.log


uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
    --mc_mae_weight 0.87 \
    --mc_lpips_weight 0.05 \
      --mc_lpips_calib_weight 0.03 \
    --mc_tv_weight 1e-3 \
    --mc_gdl_weight 0.05 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-87-5_5__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-87-5_5__depth_4__48_84_146_256__deeper_3masks.log



uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
    --mc_mae_weight 0.95 \
    --mc_lpips_weight 0.01 \
      --mc_lpips_calib_weight 0.03 \
    --mc_tv_weight 1e-3 \
    --mc_gdl_weight 0.04 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-95-1_4__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-95-1_4__depth_4__48_84_146_256__deeper_3masks.log


uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
    --mc_mae_weight 0.99 \
    --mc_lpips_weight 0.0 \
      --mc_lpips_calib_weight 0.02 \
    --mc_tv_weight 1e-3 \
    --mc_gdl_weight 0.001 \
  --mc_lpips_net alex \
  --input_extrema_prob .6\
  --input_sparse_keep_prob .3 \
  --input_decimate_trilinear_prob .1 \
--encoder_depth_profile deeper \
--unet_levels 4 --hidden_dims 48 84 146 256 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-98-0_1__depth_4__48_84_146_256__deeper_3masks \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-98-0_1__depth_4__48_84_146_256__deeper_3masks.log


uv run python studies/run_random_training_sweep.py \
--num_runs 100 \
--prune_start_after 5 \
--tb_budget_gb 80 \    
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--mc_lpips_net alex \ 
--encoder_depth_profile deeper \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_base checkpoints \                                                                                  
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/run_random_training_sweep.log 


uv run python studies/run_random_training_sweep.py \ 
--num_runs 100 \
--prune_start_after 5 \
--tb_budget_gb 80 \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--mc_lpips_net alex \
--encoder_depth_profile deeper \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_base checkpoints \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/run_random_training_sweep.log

# -----------------------------------------------------------------------------
# Top-10 unattended retraining loop (preferred replacement for manual block).
# Usage:
#   run_top10_retrain_loop
#   run_top10_retrain_loop --dry-run
#   run_top10_retrain_loop --start-index 1200
run_top10_retrain_loop() {
  uv run python studies/run_top10_retrain_loop.py \
    --schedule 2,2,3,3,5,5,5,5 \
    --start-index 1000 \
    --datasets-per-pass 10 \
    --data-folder /Users/donaldpg/synthoseis/fake_data \
    --real-train-paths /Users/donaldpg/synthoseis/real_data \
    --real-test-paths /Users/donaldpg/synthoseis/fake_data/test \
    --epoch-samples 200000 \
    --real-epoch-samples 50000 \
    --train-batches-per-epoch 100 \
    --val-batches-per-epoch 40 \
    --test-batches-per-epoch 60 \
    --batch-size 2 \
    --val-split-ratio 0.3 \
    --lr-min 5e-6 \
    --state-path checkpoints/top10_retrain_loop_state.json \
    --log-file logs/top10_retrain_loop.log \
    --max-retries 2 \
    --retry-delay-sec 300 \
    "$@"
}

# One-line replacement for the long manual sequence below:
# run_top10_retrain_loop

### copilot suggested 10 runs that it thinks could beat all the random parameter combos
# 1. Best-conservative candidate, almost identical to the current winner but with a tiny LPIPS term:
# 2. Same family, slightly wider to push capacity while still staying close to the best run:
# 3. Slightly narrower 3-level variant, which may regularize better than the current top run:
# 4. Best 4-level family, nudged toward more representational capacity:
# 5. Aggressive 4-level variant, still MAE-first but with a small LPIPS signal:

top_10_best_checkpoints = [
    "checkpoints/checkpoint_copilot_5/checkpoint_epoch_0034.pt",
    "checkpoints/checkpoint_copilot_2/checkpoint_epoch_0029.pt",
    "checkpoints/checkpoint_copilot_1/checkpoint_epoch_0033.pt",
    "checkpoints/checkpoint_copilot_4/checkpoint_epoch_0035.pt",
    "checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_epoch_0008.pt",
]

date
sleep 2h
echo "starting now" 
date
#rm -r /Users/donaldpg/synthoseis/fake_data/seismic__2026.47*__synthoseis_run_06*
cd /Users/donaldpg/synthoseis-pretrain-v2
~/synthoseis-pre-train/generate_datasets.sh -n 10 \
  --synthoseis-dir ~/synthoseis/synthoseis \
  -d ~/synthoseis/fake_data \
  --start-index 687



cd *_1
cp -p checkpoint_epoch_0095.pt checkpoint_final_model.pt
cd ../*_2
cp -p checkpoint_epoch_0095.pt checkpoint_final_model.pt
cd ../*_4
cp -p checkpoint_epoch_0095.pt checkpoint_final_model.pt
cd ../*_5
cp -p checkpoint_epoch_0095.pt checkpoint_final_model.pt

cd sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt

cd ../sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt
cd ../sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt
cd ../sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt
cd ../sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt
cd ../sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt
cd ../sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001
cp -p checkpoint_epoch_0050.pt checkpoint_final_model.pt



date
sleep 5h
echo "starting now" 
date
rm -r /Users/donaldpg/synthoseis/fake_data/seismic__2026.497*__synthoseis_run_07*
cd /Users/donaldpg/synthoseis-pretrain-v2
~/synthoseis-pre-train/generate_datasets.sh -n 10 \
  --synthoseis-dir ~/synthoseis/synthoseis \
  -d ~/synthoseis/fake_data \
  --start-index 748



uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/checkpoint_copilot_1 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 100 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 3 --hidden_dims 40 80 160 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.0 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 1e-5  \
--resume checkpoints/checkpoint_copilot_1/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/checkpoint_copilot_2 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 100 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 3 --hidden_dims 40 88 176 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.001 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 1e-5  \
--resume checkpoints/checkpoint_copilot_2/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/checkpoint_copilot_4 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 100 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 40 80 160 256 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.0 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 1e-5  \
--resume checkpoints/checkpoint_copilot_4/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/checkpoint_copilot_5 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 100 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 44 88 176 256 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.001 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 1e-5 2>&1 \
--resume checkpoints/checkpoint_copilot_5/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python studies/prune_pt_in_best_val_folders.py --apply
###

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 40 74 138 256 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.0 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 5e-5  \
--resume checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 32 64 128 256 --encoder_depth_profile baseline \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.990 --mc_lpips_weight 0.000 --mc_tv_weight 0.000 --mc_gdl_weight 0.010 \
--lr_min 5e-6 --lr 5e-5  \
--resume checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python studies/prune_pt_in_best_val_folders.py --apply

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 3 --hidden_dims 40 80 160 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.001 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 5e-5  \
--resume checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 3 --hidden_dims 32 64 128 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.0 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 5e-5  \
--resume checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python studies/prune_pt_in_best_val_folders.py --apply

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 40 74 138 256 --encoder_depth_profile deeper \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.995 --mc_lpips_weight 0.005 --mc_tv_weight 0.001 --mc_gdl_weight 0 \
--lr_min 5e-6 --lr 5e-5 2>&1 \
--resume checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log

uv run python -u train_cli.py --loss multi_component --mc_lpips_net alex \
--output_dir checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
--real_train_paths /Users/donaldpg/synthoseis/real_data \
--real_test_paths /Users/donaldpg/synthoseis/fake_data/test \
--epoch_samples 200000 \
--real_epoch_samples 50000 \
--test_batches_per_epoch 60 \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--val_split_ratio 0.3 --unet_levels 4 --hidden_dims 40 74 138 256 --encoder_depth_profile deepest \
--mc_mse_weight 0 --mc_pmse_weight 0 --mc_mae_weight 0.999 --mc_lpips_weight 0.0 --mc_tv_weight 0.0 --mc_gdl_weight 0.001 \
--lr_min 5e-6 --lr 5e-5 2>&1 \
--resume checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_final_model.pt \
| tee -a logs/run_random_training_colpilot_sweep.log


##########################################################################################

uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight 0.000000 \
  --mc_pmse_weight 0.000000 \
  --mc_mae_weight 0.980000 \
  --mc_lpips_weight 0.010000 \
  --mc_gdl_weight 0.000000 \
  --mc_tv_weight 0.010000 \
  --mc_lpips_net alex \
--input_extrema_prob 0.6 --input_sparse_keep_prob 0.3 --input_decimate_trilinear_prob 0.1 \
--encoder_depth_profile deeper --unet_levels 3 \
--hidden_dims 40 80 160 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1 \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.025 --lr_poly_power 0.9 \
--lr_min 5e-06 --lr 1e-05 \
--output_dir checkpoints/sweep_20260620_055743_r005_u3_h40-80-160_lp0p010_tv0p010 \
--data_folder /Users/donaldpg/synthoseis/fake_data \
--tb_image_epochs 1 5 8 9 10 \
--batch_size 2 \
2>&1 | tee -a logs/run_random_training_sweep.log 


### mae only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 1.0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 4 --hidden_dims 48 78 128 208 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_78_128_208 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_4__48_78_128_208.log


#####################################################

### mae only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 0.9 \
  --mc_lpips_weight .1 \
  --mc_tv_weight 1e-2 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-99-1__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-99-1__depth_3.log

### pmse only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight 0.9 \
  --mc_mae_weight .0 \
  --mc_lpips_weight .1 \
  --mc_tv_weight 1e-2 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-99-0-1__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-99-0-1__depth_3.log

### mse only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight 0.9 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .0 \
  --mc_lpips_weight .1 \
  --mc_tv_weight 1e-2 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__99-0-0-1__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__99-0-0-1__depth_3.log

### lpips only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .0 \
  --mc_lpips_weight 1.0 \
  --mc_tv_weight 1e-2 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-0-100__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-0-100__depth_3.log

#####################################################

#####################################################

### mse only
uv run python -u train_cli.py \
--epochs 25 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight 1.0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__100-0-0-0__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__100-0-0-0__depth_3.log
### pmse only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight 1.0 \
  --mc_mae_weight .0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-100-0-0__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-100-0-0__depth_3.log
### mae only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight 1.0 \
  --mc_lpips_weight .0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-100-0__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-100-0__depth_3.log
### lpips only
uv run python -u train_cli.py \
--epochs 10 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 100 --val_batches_per_epoch 40 \
--loss multi_component \
  --mc_mse_weight .0 \
  --mc_pmse_weight .0 \
  --mc_mae_weight .0 \
  --mc_lpips_weight 1.0 \
  --mc_tv_weight 1e-3 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 2.5e-5 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_mse_pmse_mae_lpips__0-0-0-100__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_mse_pmse_mae_lpips__0-0-0-100__depth_3.log

#####################################################

uv run python -u train_cli.py \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 40 --val_batches_per_epoch 25 \
--loss multi_component \
--mc_mse_weight .10 \
  --mc_pmse_weight .40 \
  --mc_mae_weight .40 \
  --mc_lpips_weight .10 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 5e-6 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_sliding_stats_mse_pmse_mae_lpips__10-40-40-10__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_multi_datasets_sliding_stats_mse_pmse_mae_lpips__10-40-40-10__depth_3.log


uv run python -u train_cli.py \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 40 --val_batches_per_epoch 25 \
--loss multi_component \
--mc_mse_weight .30 \
  --mc_pmse_weight .60 \
  --mc_mae_weight .05 \
  --mc_lpips_weight .05 \
  --mc_lpips_net alex \
--unet_levels 3 --hidden_dims 32 64 128 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 5e-6 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_sliding_stats_mse_pmse_mae_lpips__30-60-5-5__depth_3 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_multi_datasets_sliding_stats_mse_pmse_mae_lpips__30-60-5-5__depth_3.log

uv run python -u train_cli.py \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 40 --val_batches_per_epoch 25 \
--loss multi_component \
--mc_mse_weight .30 \
  --mc_pmse_weight .60 \
  --mc_mae_weight .05 \
  --mc_lpips_weight .05 \
  --mc_lpips_net alex \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 5e-6 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_sliding_stats_mse_pmse_mae_lpips__30-60-5-5 \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_multi_datasets_sliding_stats_mse_pmse_mae_lpips__30-60-5-5.log

uv run python -u train_cli.py \
--loss_type sliding_stats --sliding_stats_window 9 9 9 \
--sliding_stats_all_voxels --sliding_stats_mean_weight 0.0 --sliding_stats_std_weight 0.0 \
--sliding_stats_mae_weight 0.0 --sliding_stats_mse_weight 1.0  \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 40 --val_batches_per_epoch 25 \
--grad_accum_steps 1 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 5e-6 --lr 5.000e-05 \
--output_dir checkpoints/checkpoints_sliding_stats_mse --print_model_summary \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a logs/train_multi_datasets_sliding_stats_mse.log

uv run python -u train.py \
--loss_type sliding_stats --sliding_stats_window 9 9 9 \
--sliding_stats_all_voxels --sliding_stats_mean_weight 4.0 --sliding_stats_std_weight 1.3 \
--sliding_stats_mae_weight 1.3 --sliding_stats_mse_weight 0.8  \
--epochs 60 --val_split_ratio 0.3 --batch_size 2  \
--train_batches_per_epoch 40 --val_batches_per_epoch 25 \
--grad_accum_steps 15 --grad_clip_norm 1.0 \
--ema_decay 0.995 --ema_update_every 1  \
--lr_schedule poly --lr_warmup_epochs 3 --lr_warmup_start_factor 0.1 \
--lr_poly_power 0.9 --lr_min 5e-6 --lr 5.000e-05 \
--output_dir checkpoints_sliding_stats6b --print_model_summary \
--data_folder /Users/donaldpg/synthoseis/fake_data  \
2>&1 | tee -a train_multi_datasets_sliding_stats6b.log

