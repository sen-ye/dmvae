#!/bin/bash

# Common variables for distributed training
export num_nodes=${NNODES:-1}
export node_rank=${NODE_RANK:-0}
export master_addr=${MASTER_ADDR:-"localhost"}
export master_port=${MASTER_PORT:-29511}
export nproc_per_node=${NPROC_PER_NODE:-8}

exp_name="train_dmd"
log_dir="./exps"
lpips_path="./ckpts/ckpt_vae/vgg.pth"
dataset_path="./data/ImageNet"
vae_ckpt_path="./ckpts/vae_ckpt" # pretrained VAE checkpoint
model_ckpt_path="./ckpts/model_ckpt" # teacher diffusion model checkpoint
vae_train_every=5
dmd_weight=10.0 # 10/20 is recommended for DMD training
dmd_cfg_scale=5.0 # 5 is recommended for DMD training

torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train_dmd.py \
    --exp_name $exp_name \
    --log_dir $log_dir \
    --epoch 100 \
    --local_bs 16 \
    --workers 4 \
    --lr 2e-5 \
    --diff_lr 2e-5 \
    --disc_lr 2e-5 \
    --disc_start_step 5000 \
    --dataset_path $dataset_path \
    --lpips_path $lpips_path \
    --vae_ckpt_path $vae_ckpt_path \
    --model_ckpt_path $model_ckpt_path \
    --vae_train_every $vae_train_every \
    --dmd_weight $dmd_weight \
    --dmd_cfg_scale $dmd_cfg_scale