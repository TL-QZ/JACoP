# User-defined log file name
log_name="JACoP_SDD_$(date +'%Y%m%d_%H%M%S')"
log_file="test_log/${log_name}.log"
mkdir -p test_log

CUDA_VISIBLE_DEVICES=7 python test.py \
    --model MRF \
    --root data/SDD\
    --ckpt_path "ckpts/sdd.ckpt" \
    --batch_size 1 \
    --num_workers 8 \
    --pin_memory True \
    --persistent_workers True \
    --devices 1 \
    --overwrite \
    --sampling_mode gibbs \
    --save_output \
    --save_path outputs/MRF_SDD \
    --devices 1 \
    --collision_thresh 5 | tee -a "$log_file"
