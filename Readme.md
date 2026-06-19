# JACoP: Joint Alignment for Compliant Multi-Agent Prediction

Official implementation of **JACoP Joint Alignment for Compliant Multi-Agent Predictio**

[\[arXiv\]](https://arxiv.org/abs/2605.11385) [\[CVPR 2026 paper page\]](https://openaccess.thecvf.com/content/CVPR2026F/html/Liu_JACoP_Joint_Alignment_for_Compliant_Multi-Agent_Prediction_CVPRF_2026_paper.html) [\[Project Website\]](https://tl-qz.github.io/jacop.github.io/)



## Environment

Create a Python environment with PyTorch, PyTorch Geometric, and PyTorch Lightning versions compatible with your CUDA driver. The code imports the following packages:

```bash
pip install numpy pandas scipy matplotlib tqdm pillow opencv-python scikit-image shapely potrace pickle5 imageio torchvision tensorboard lightning
```

Install PyTorch and PyTorch Geometric from their official instructions for your CUDA version. For example, use the PyTorch selector for `torch`/`torchvision`, then install `torch-geometric` and its compiled extensions with wheels matching the installed PyTorch/CUDA build.

## Data Folder

`data/` is intentionally gitignored. Download or prepare the datasets so they match these layouts.

ETH/UCY leave-one-out splits:

```text
data/eth_ucy/
  maps/
    maps/{eth,hotel,univ,zara1,zara2}.png
    homo_mats/{eth,hotel,univ,zara1,zara2}_H.txt
  eth/{train,val,test}/
  hotel/{train,val,test}/
  univ/{train,val,test}/
  zara1/{train,val,test}/
  zara2/{train,val,test}/
```

SDD:

```text
data/SDD/
  train/train_trajnet.pkl
  val/val_trajnet.pkl
  test/test_trajnet.pkl
  semantic_maps/*_mask.png
```

Processed dataset caches are created automatically under:

```text
<dataset_root>/processed/{train,val,test}/
```

If you regenerate anchors after processed files already exist, delete the corresponding `processed/` directory before training or testing, because anchor labels and environment-compliance scores are stored in the processed cache.

## Step 1: Generate Anchors

JACoP requires one anchor file at:

```text
<dataset_root>/anchors/trajectory_prototypes.npy
```

Use the public anchor CLI:

```bash
# ETH/UCY: generate all five leave-one-out split anchors
python prepare/generate_anchors.py \
  --dataset eth_ucy \
  --data-root data/eth_ucy \
  --splits eth hotel univ zara1 zara2

# SDD
python prepare/generate_anchors.py \
  --dataset sdd \
  --data-root data/SDD
```

Useful options:

```bash
--num-outputs 20      # number of trajectory prototypes
--num-samples 20      # compatibility alias for --num-outputs
--output-root /tmp/x  # optional non-destructive output root for smoke tests
--overwrite           # replace existing trajectory_prototypes.npy files
```

The older `prepare/prep_anchor.py` entrypoint is kept for compatibility, but `prepare/generate_anchors.py` is the recommended interface because it does not require editing JSON config files.

## Step 2: Train

Training automatically preprocesses raw data into the `processed/` cache when needed. Validation and testing batch size are forced to `1` inside `train.py`.

Run the provided scripts:

```bash
# ETH/UCY: trains leave-one-out splits on a range of GPU ids
bash script/MRF_ETH_UCY/train/train.sh 0 5

# SDD
bash script/MRF_SDD/train/train.sh
```

Direct ETH/UCY example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model MRF \
  --exp_name JACoP_ETH \
  --root data/eth_ucy/eth \
  --dataset eth_ucy \
  --train_batch_size 32 \
  --val_batch_size 1 \
  --test_batch_size 1 \
  --num_workers 8 \
  --devices 1 \
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
  --pairwise_potential_type distance \
  --distance_type cosine \
  --env_hist_fuse \
  --unary_recon_loss wta \
  --qc_encoding_only \
  --apply_env_filtering \
  --collision_thresh 0.2 \
  --apply_collision_filtering
```

Direct SDD example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model MRF \
  --exp_name JACoP_SDD \
  --root data/SDD \
  --dataset sdd \
  --train_batch_size 128 \
  --val_batch_size 1 \
  --test_batch_size 1 \
  --num_workers 8 \
  --devices 1 \
  --num_historical_steps 8 \
  --num_future_steps 12 \
  --hidden_dim 64 \
  --pl2a_radius 0 \
  --a2a_radius 50 \
  --bp_iter 3 \
  --num_modes 20 \
  --lr 0.001 \
  --unary_lambda 1.0 \
  --unary_only_until 15 \
  --max_epochs 500 \
  --T_max 500 \
  --pairwise_loss_fn focal \
  --pairwise_potential_type distance \
  --distance_type cosine \
  --env_hist_fuse \
  --unary_recon_loss wta \
  --qc_encoding_only \
  --apply_env_filtering \
  --collision_thresh 0.5 \
  --apply_collision_filtering
```

Checkpoints and TensorBoard logs are written under `lightning_logs/MRF/<exp_name>/`.

## Test

Pretrained checkpoints are included:

```text
ckpts/eth.ckpt
ckpts/hotel.ckpt
ckpts/univ.ckpt
ckpts/zara1.ckpt
ckpts/zara2.ckpt
ckpts/sdd.ckpt
```

Run all ETH/UCY splits:

```bash
bash script/MRF_ETH_UCY/test/test.sh
```

Run SDD:

```bash
bash script/MRF_SDD/test/test.sh
```

Direct ETH example:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model MRF \
  --dataset eth_ucy \
  --root data/eth_ucy/eth \
  --ckpt_path ckpts/eth.ckpt \
  --batch_size 1 \
  --num_workers 8 \
  --devices 1 \
  --overwrite \
  --sampling_mode gibbs \
  --save_output \
  --save_path outputs/MRF_gibbs \
  --collision_thresh 0.2
```

Direct SDD example:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model MRF \
  --dataset sdd \
  --root data/SDD \
  --ckpt_path ckpts/sdd.ckpt \
  --batch_size 1 \
  --num_workers 8 \
  --devices 1 \
  --overwrite \
  --sampling_mode gibbs \
  --save_output \
  --save_path outputs/MRF_SDD \
  --collision_thresh 5
```

When `--save_output` is set, predictions are saved as pickle files under the selected `--save_path`.

## Repository Structure

```text
train.py, test.py              Training and evaluation entrypoints
predictors/MRF.py              JACoP/MRF model
datasets/, datamodules/        ETH/UCY and SDD data processing
prepare/                       Anchor generation utilities
script/                        Reproducible train/test shell scripts
ckpts/                         Released checkpoints
outputs/                       Example saved predictions
```

##  If you find this repository useful, please cite:

```bibtex
@inproceedings{liu2026jacop,
  title={JACoP: Joint Alignment for Compliant Multi-Agent Prediction},
  author={Liu, Qingze Tony and Mrdovic, Alen and Li, Danrui and Schwartz, Mathew and Yoon, Sejong and Kapadia, Mubbasir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={910--919},
  year={2026}
}
```
