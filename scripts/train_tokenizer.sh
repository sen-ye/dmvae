#!/bin/bash

# Common variables for distributed training
export num_nodes=${NNODES:-1}
export node_rank=${NODE_RANK:-0}
export master_addr=${MASTER_ADDR:-"localhost"}
export master_port=${MASTER_PORT:-29511}
export nproc_per_node=${NPROC_PER_NODE:-8}



exp_name="train_tokenizer"
log_dir=./exps/
freeze_encoder=True
freeze_mlp=False
lpips_path=./ckpt_vae/vgg.pth
dataset_path=./data/ImageNet
epoch=10 # 10 is recommended for tokenizer pretraining, 50 is recommended for decoder fine-tuning

torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train_tokenizer.py \
    --exp_name $exp_name \
    --log_dir $log_dir \
    --epoch $epoch \
    --local_bs 16 \
    --workers 4 \
    --lr 1e-4 \
    --disc_lr 1e-4 \
    --disc_start_step 5000 \
    --dataset_path $dataset_path \
    --freeze_encoder $freeze_encoder \
    --freeze_mlp $freeze_mlp \
    --lpips_path $lpips_path \