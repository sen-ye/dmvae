import os
import random
import torch
import torch.distributed as dist
import numpy as np
from diffusion.lightningdit.lightningdit import LightningDiT_models, LightningDiT
from diffusion.transport import create_transport, Sampler
from tqdm import tqdm
from PIL import Image
import math
import torch_fidelity # type: ignore
from tap import Tap
from typing import Literal, Optional
from models.vae import VAE


class Args(Tap):
    exp_name: str = "sample_50k"
    mode: Literal["ODE", "SDE"] = "SDE"
    model: str = "LightningDiT-XL/1"
    sample_dir: str = f""
    per_proc_batch_size: int = 25
    num_fid_samples: int = 50000
    image_size: int = 256
    num_classes: int = 1000
    cfg_scale: float = 1.0
    num_sampling_steps: int = 250
    global_seed: int = 0
    tf32: bool = True
    ckpt: Optional[str] = ""
    ckpt_model_name: str = "ema"
    additional_model: str = "LightningDiT-XL/1"
    additional_model_ckpt: str = ""
    additional_model_name: str = "ema"
    latent_mean: float = 0
    latent_scale: float = 1
    z_channels: int = 32
    model_size: str = "large"
    vae_ckpt_path: str = ""
    vae_ckpt_ema: bool = False
    path_type: str = "Linear"
    prediction: str = "velocity"
    loss_weight: Optional[str] = None
    sample_eps: float = 0.0
    train_eps: float = 0.0
    sampling_method: str = "Euler"
    atol: float = 1e-6
    rtol: float = 1e-3
    reverse: bool = False
    likelihood: bool = False
    diffusion_form: str = "sigma"
    diffusion_norm: float = 1.0
    last_step: Optional[str] = "Mean"
    last_step_size: float = 0.04
    fid_statistics_file: Optional[str] = f""
    t_min: float = 0.0
    t_max: float = 1.0
    inception_weights_path: str = f""
    time_dist_shift: float = 1.0


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def sample(args: Args, device, rank, world_size):
    latent_size = args.image_size // 16
    model = LightningDiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state_dict[args.ckpt_model_name]
    model.load_state_dict(state_dict)
    model.eval()

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        time_dist_shift=args.time_dist_shift,
    )
    sampler = Sampler(transport)
    if args.mode == "ODE":
        sample_fn = sampler.sample_ode(
                    sampling_method=args.sampling_method,
                    num_steps=args.num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                    reverse=args.reverse,
        )
    else:
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    vae_ckpt_full_path = args.vae_ckpt_path
    if rank == 0:
        print(f"Loading VAE from {vae_ckpt_full_path}")
    vae = VAE(
        model_size=args.model_size,
        z_channels=args.z_channels,
    )
    vae.load_pretrained(vae_ckpt_full_path, ema=args.vae_ckpt_ema)
    vae = vae.eval().to(device)
    for p in vae.parameters(): p.requires_grad = False
    dist.barrier()
    if rank == 0:
        print("VAE loaded")

    latent_mean = args.latent_mean
    latent_scale = args.latent_scale
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    sample_folder_dir = os.path.join(args.sample_dir, args.exp_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    world_size = dist.get_world_size()
    label_list = list(range(args.num_classes))
    num_samples_per_class = args.num_fid_samples // args.num_classes
    label_list = label_list * num_samples_per_class
    num_samples_per_rank = len(label_list) // world_size
    start_idx = num_samples_per_rank * rank
    end_idx = start_idx + num_samples_per_rank
    label_list_this_rank = label_list[start_idx:end_idx]
    assert num_samples_per_rank % args.per_proc_batch_size == 0, "num_samples_per_rank must be divisible by per_proc_batch_size"
    num_iterations = int(math.ceil(num_samples_per_rank / args.per_proc_batch_size))
    pbar = tqdm(range(num_iterations))
    total = 0
    save_latents = []
    for iter_idx in pbar:
        total += (n * world_size)
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.tensor(label_list_this_rank[iter_idx * n: (iter_idx + 1) * n], device=device)

        model_kwargs = dict(y=y)
        model_fn = model.forward
        samples = sample_fn(z, model_fn, **model_kwargs)[-1]

        p = 1
        n_batch, c, h, w = samples.shape
        samples = samples.reshape(shape=(n_batch, c, h, p, h, p))
        samples = torch.einsum('nchpwq->nhwpqc', samples)
        samples = samples.reshape(shape=(n_batch, h * w, p * p * c))
        samples = samples / latent_scale + latent_mean
        save_latents.append(samples.detach().to("cpu", non_blocking=True))
        samples = vae.decode(samples).float()

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8, non_blocking=True).numpy()

        for local_idx, sample in enumerate(samples):
            index = local_idx * world_size + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")


    model = model.to("cpu")
    vae = vae.to("cpu")
    del model
    del vae
    torch.cuda.empty_cache()
    dist.barrier()

    with torch.autocast("cuda", enabled=False):
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder_dir,
            input2=None, # TODO: maybe update this 
            samples_shuffle=True,
            samples_find_deep=True,
            samples_find_ext='png,jpg,jpeg',
            samples_ext_lossy='jpg,jpeg',
            fid_statistics_file=args.fid_statistics_file, # TODO: maybe update this
            cuda=True,
            batch_size=1024,
            isc=True,
            fid=True,

            kid=False,
            kid_subsets=100,
            kid_subset_size=1000,

            ppl=False,
            prc=False,
            ppl_epsilon=1e-4 or 1e-2,
            ppl_sample_similarity_resize=64,
            feature_extractor='inception-v3-compat',
            feature_layer_isc='logits_unbiased',
            feature_layer_fid='2048',  # '64'
            feature_layer_kid='2048',  # '64'
            feature_extractor_weights_path=args.inception_weights_path, # TODO: update inception path
            verbose=True,

            save_cpu_ram=False,  # using num_workers=0 for any dataset input1 input2
            rng_seed=0,  # FID isn't sensitive to this
        )
    fid = metrics_dict["frechet_inception_distance"]
    inception_score = metrics_dict["inception_score_mean"]
    print("sample folder: ", sample_folder_dir)
    print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}")


def seed_everything(seed):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args: Args):
    """Run sampling with distributed decoding."""
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    seed = 42 + rank
    seed_everything(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    sample(args, device, rank, world_size)

    dist.destroy_process_group()



# torchrun --nproc_per_node=8 --master_port=29511 sample_50k.py
if __name__ == "__main__":
    args = Args(explicit_bool=True).parse_args(known_only=True)
    main(args)
