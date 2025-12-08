#!/bin/bash
# Common variables for distributed training
export num_nodes=${NNODES:-1}
export node_rank=${NODE_RANK:-0}
export master_addr=${MASTER_ADDR:-"localhost"}
export master_port=${MASTER_PORT:-29511}
export nproc_per_node=${NPROC_PER_NODE:-8}



exp_name="train_diffusion"
log_dir="./exps"
dataset_path="./data/ImageNet"
vae_ckpt_path="./ckpts/" # should point to the checkpoint directory generated in the tokenizer pretraining stage
model="LightningDiT-XL/1"
latent_mean=0.0 # should set to the mean of the VAE latent space 
latent_scale=1.0 # should set to the scale of the VAE latent space
use_checkpoint=True
local_bs=64 # ensure global_bs is 2048
lr=2e-4 # follow VA-VAE

torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train_diffusion.py \
    --exp_name $exp_name \
    --log_dir $log_dir \
    --dataset_path $dataset_path \
    --vae_ckpt_path $vae_ckpt_path \
    --model $model \
    --latent_mean $latent_mean \
    --latent_scale $latent_scale \
    --lr $lr \
    --local_bs $local_bs \
    --use_checkpoint $use_checkpoint \
    $@