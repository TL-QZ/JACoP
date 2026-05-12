python test.py \
--model MRF --root data/SDD --ckpt_path "lightning_logs/MRF/MRF_SDD_Train_0122/SDD/version_0/checkpoints/epoch=99-step=60100.ckpt" --batch_size 1 --num_workers 8 --pin_memory True --persistent_workers True --devices 1 --overwrite --sampling_mode gibbs --save_output --save_path outputs/MRF_SDD --devices 1 --collision_thresh 0.5
