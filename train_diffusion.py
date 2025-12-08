import os
import torch
import torch.distributed as tdist
from torch.nn.attention import SDPBackend, sdpa_kernel
import utils.dist as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
from contextlib import nullcontext
from collections import OrderedDict
from copy import deepcopy
from time import time
import wandb
from utils.build_dataset import build_dataset
from diffusion.lightningdit.lightningdit import LightningDiT_models, LightningDiT
from diffusion.transport import create_transport, Sampler, Transport
from models.vae import VAE
from tap import Tap
from typing import Optional, Union


class Args(Tap):
    project_name: str = "DMVAE"
    exp_name: str = "train"
    dataset_type: str = "imagenet"
    dataset_path: str = ""
    log_dir: str = ""
    local_bs: int = 32
    global_bs: int = 256
    workers: int = 4
    same_seed_for_all_ranks: bool = False
    hflip: bool = True
    model: str = "LightningDiT-XL/1"
    image_size: int = 256
    num_classes: int = 1000
    epochs: int = 800
    global_seed: int = 0
    log_every: int = 100
    ckpt_every: int = 20_000
    sample_every: int = 10_000
    cfg_scale: float = 4.0
    wandb: bool = True
    latent_mean: float = 0.
    latent_scale: float = 1.
    path_type: str = "Linear"
    prediction: str = "velocity"
    lr: float = 1e-4
    loss_weight: str = None
    sample_eps: float = 0.0
    train_eps: float = 0.0
    seed: int = 42
    # VAE related
    z_channels: int = 32
    patch_size: int = 16
    model_size: str = "large"    
    vae_ckpt_path: str = ""
    # training related
    tf32: bool = True
    bf16: bool = True
    flash_attn: bool = True
    seed: int = 42
    use_checkpoint: bool = True
    class_dropout_prob: float = 0.1

    @staticmethod
    def set_tf32(tf32: bool = True):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            seed = self.seed + dist.get_rank() * 10000
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:  # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:  # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f'WARNING: {k} not in args')



#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def wandb_log(data, step=None, log_ferq=None, commit=None, sync=None):
    if not dist.is_master():
        return
    if step is not None and log_ferq is not None:
        if step % log_ferq == 0:
            wandb.log(data, step=step, commit=commit, )
        else:
            return
    else:
        wandb.log(data, step=step, commit=commit,)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    tdist.destroy_process_group()



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args: Args):
    # Setup DDP:
    device = int(os.environ.get("LOCAL_RANK"))
    rank = dist.get_rank()

    # Setup an experiment folder:
    if rank == 0:
        checkpoint_dir = os.path.join(args.log_dir, 'ckpts')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.wandb:
            wandb.init(
                project=args.project_name,
                resume='auto',
                save_code=True,
                id=args.exp_name,
                name=args.exp_name,
                config=args.state_dict()
            )
            wandb.run.log_code(".", include_fn=lambda x: x.endswith('.py'))

    # log args
    print(f"Args: {args.state_dict()}")

    # Create model:
    assert args.image_size % 16 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 16

    print(f"[LightningDiT] loading model: {args.model}")
    model = LightningDiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_checkpoint=args.use_checkpoint,
        class_dropout_prob=args.class_dropout_prob
    )
    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    model_wo_ddp = model
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )  # default: velocity; 
    transport_sampler = Sampler(transport)


    print(f"[LightningDiT] parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0, fused=True)

    train_data_info = build_dataset(
        image_size=args.image_size,
        mode="train",
        dataset_path=args.dataset_path,
        hflip=args.hflip,
        workers=args.workers,
        same_seed_for_all_ranks=args.same_seed_for_all_ranks,
        global_bs=args.global_bs,
        local_bs=args.local_bs,
        get_different_generator_for_each_rank=args.get_different_generator_for_each_rank
    )
    # load vae
    print(f"[VAE] loading checkpoint from {args.vae_ckpt_path}")
    vae = VAE(model_size=args.model_size, z_channels=args.z_channels, patch_size=args.patch_size,)
    vae.load_pretrained(args.vae_ckpt_path)
    vae = vae.eval().to(device)
    for p in vae.parameters(): p.requires_grad = False
    print(f"[VAE] loaded")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0

    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(args.num_classes, size=(2,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 32, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale, standard_cfg=True)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    
    latent_mean = args.latent_mean
    latent_scale = args.latent_scale


    print(f"Training for {args.epochs} epochs...")
    attn_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]) if args.flash_attn else nullcontext()
    with attn_context:
        for epoch in range(args.epochs):
            train_data_info.set_epoch(epoch)
            print(f"Beginning epoch {epoch}...")
            num_batches = train_data_info.num_batches
            for cur_iter in range(num_batches):
                x, y = next(train_data_info.dataloader)
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        x = vae.encode(x)
                    x = (x - latent_mean) * latent_scale
                    c = x.shape[-1]
                    h = w = int(x.shape[1] ** 0.5)
                    assert h * w == x.shape[1]
                    p = 1
                    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
                    x = torch.einsum('nhwpqc->nchpwq', x)
                    x = x.reshape(shape=(x.shape[0], c, h * p, h * p))


                    model_kwargs = dict(y=y)
                    t, loss_dict = transport.training_losses(model, x, model_kwargs)

                loss = loss_dict["loss"].mean().float()
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                update_ema(ema, model.module)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    tdist.all_reduce(avg_loss, op=tdist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {grad_norm:.4f}")

                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save SiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args.state_dict(),
                            "steps": train_steps,
                            "epoch": epoch,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()
                
                if (train_steps % args.sample_every == 0 and train_steps > 0) or train_steps == 1:
                    print("Generating EMA samples...")
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    ema.eval()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                        samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                        dist.barrier()

                        if use_cfg: #remove null samples
                            samples, _ = samples.chunk(2, dim=0)


                        p = 1
                        n,c,h,w = samples.shape
                        samples = samples.reshape(shape=(n, c, h, p, h, p))
                        samples = torch.einsum('nchpwq->nhwpqc', samples)
                        samples = samples.reshape(shape=(n, h*w, p*p*c)) / latent_scale + latent_mean

                        samples = vae.decode(samples).float()

                    out_samples = torch.zeros((n*dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    tdist.all_gather_into_tensor(out_samples, samples)
                    if args.wandb:
                        wandb_log({"samples": out_samples}, train_steps)
                    print("Generating EMA samples done.")

    print("Done!")
    cleanup()


def init_args() -> Args:
    args: Args = Args(explicit_bool=True).parse_args(known_only=True)
    dist.init_distributed_mode(f"./local_output/{args.exp_name}")
    args.device = dist.get_device()
    args.global_bs = args.local_bs * dist.get_world_size()
    args.seed_everything()
    args.set_tf32(args.tf32)
    return args


if __name__ == "__main__":
    args:Args = init_args()
    main(args)