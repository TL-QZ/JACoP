#! /bin/bash

# User-defined log file name
log_name="JACoP_0202_ETH_UCY$(date +'%Y%m%d_%H%M%S')"
log_file="test_log/${log_name}.log"
mkdir -p test_log

device_id=6 # Specify the GPU device ID to use

# List of ETH-UCY splits
splits=("eth" "hotel" "zara1" "zara2" "univ")
ckpt_paths=(
    "ckpts/eth.ckpt"
    "ckpts/hotel.ckpt"
    "ckpts/zara1.ckpt"
    "ckpts/zara2.ckpt"
    "ckpts/univ.ckpt"
)

separator="=============================="

for i in "${!splits[@]}"; do
    split="${splits[$i]}"
    ckpt_path="${ckpt_paths[$i]}"

    echo "${separator}" | tee -a "$log_file"
    echo "Testing split: $split" | tee -a "$log_file"
    echo "Checkpoint: $ckpt_path" | tee -a "$log_file"
    echo "${separator}" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES=$device_id python test.py \
        --model MRF \
        --root data/eth_ucy/${split} \
        --ckpt_path "${ckpt_path}" \
        --batch_size 1 \
        --num_workers 8 \
        --pin_memory True \
        --persistent_workers True \
        --overwrite \
        --sampling_mode gibbs \
        --save_output \
        --collision_thresh 0.2 \
        --save_path outputs/MRF_gibbs \
        --devices 1 | tee -a "$log_file"
done
