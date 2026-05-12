#!/bin/bash
# Usage: bash script/MRF/train/train_dev.sh [START_DEVICE] [END_DEVICE]
# Example: bash script/MRF/train/train_dev.sh 0 3

datasets=(eth hotel univ zara1 zara2)

START_DEVICE=${1:-0}
END_DEVICE=${2:-5}  # exclusive, so default is 0 4 (devices 0,1,2,3)

device=0
for i in "${!datasets[@]}"; do
    dev=$((START_DEVICE + i))
    if [ "$dev" -ge "$END_DEVICE" ]; then
        break
    fi
    CUDA_VISIBLE_DEVICES=$dev python train.py \
        --model MRF \
        --exp_name JACoP_0211_ETH_UCY_w_aug_detaches_in_loss_wta_w_regularization \
        --root data/eth_ucy/${datasets[$i]} \
        --train_batch_size 32 \
        --val_batch_size 1 \
        --test_batch_size 1 \
        --num_workers 8 \
        --devices 1 \
        --dataset eth_ucy \
        --num_historical_steps 8 \
        --num_future_steps 12 \
        --hidden_dim 64 \
        --pl2a_radius 0 \
        --a2a_radius 2.5 \
        --bp_iter 3 \
        --num_modes 20 \
        --lr 0.0001 \
        --unary_only_until 15 \
        --max_epochs 100 \
        --T_max 100 \
        --pairwise_loss_fn focal \
        --pairwise_potential_type distance\
        --distance_type cosine \
        --env_hist_fuse \
        --unary_recon_loss wta \
        --qc_encoding_only \
        --apply_env_filtering \
        --collision_thresh 0.2 \
        --apply_collision_filtering &
done

wait