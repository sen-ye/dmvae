#!/bin/bash
# Common variables for distributed training
export num_nodes=${NNODES:-1}
export node_rank=${NODE_RANK:-0}
export master_addr=${MASTER_ADDR:-"localhost"}
export master_port=${MASTER_PORT:-29511}
export nproc_per_node=${NPROC_PER_NODE:-8}


ckpt=ckpt # put diffusion ckpt here
ckpt_model_name="ema"
vae_ckpt_path=vae_ckpt_path # put vae ckpt here
vae_ckpt_ema=True
latent_mean=0.0685
latent_scale=0.1763
fid_statistics_file=./ckpt_vae/adm_in256_stats.npz
inception_weights_path=./ckpt_vae/inception-2015-12-05-6726825d.pth
sample_dir=./gen_samples/
time_dist_shift=2.5


torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$master_addr \
    --master_port=$master_port \
    sample_50k.py \
    --exp_name "sample50k" \
    --vae_ckpt_path $vae_ckpt_path \
    --vae_ckpt_ema $vae_ckpt_ema \
    --latent_mean $latent_mean \
    --latent_scale $latent_scale \
    --ckpt $ckpt \
    --ckpt_model_name $ckpt_model_name \
    --time_dist_shift $time_dist_shift