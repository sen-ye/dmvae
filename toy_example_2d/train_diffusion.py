"""
Training script for LightningDiT on GMM data.
"""
import torch
import torch.distributed as tdist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP
import utils.dist as dist
import numpy as np
import random
from contextlib import nullcontext
from collections import OrderedDict
from time import time
import wandb
import logging
import os
import matplotlib.pyplot as plt
from tap import Tap
from typing import Optional, Union, List
from diffusion.lightningdit.lightningdit import LightningDiT_models
from diffusion.transport import create_transport, Sampler
from sshpae import SShapeDistribution2D



class Args(Tap):
    local_debug: bool = False
    project_name: str = "toy_dmd"
    exp_name: str = "train_gmm"
    base_dir: str = f""
    local_bs: int = 128
    global_bs: int = 256
    model: str = "LightningDiT-Mini/1"
    num_classes: int = 1
    epochs: int = 10
    global_seed: int = 0
    log_every: int = 100
    ckpt_every: int = 2_000
    sample_every: int = 1_000
    ode_num_steps: int = 25
    cfg_scale: float = 4.0
    wandb: bool = True
    path_type: str = "Linear"
    prediction: str = "velocity"
    lr: float = 1e-4
    loss_weight: str = None
    sample_eps: float = 0.0
    train_eps: float = 0.0
    seed: int = 42
    # training related
    tf32: bool = True
    bf16: bool = True
    flash_attn: bool = True
    # GMM related
    gmm_num_components: int = 1  # number of Gaussian components in the mixture
    gmm_dim: int = 2  # dimension of each Gaussian component
    plot_samples: bool = True


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

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:  # these are not serializable
                d[k] = getattr(self, k)
        return d
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def plot_gmm_samples(generated_samples: torch.Tensor, gmm: SShapeDistribution2D,
                     path: str, title: str = None, n_real_samples: int = 1000):
    """
    可视化扩散模型生成的样本。
    
    Args:
        generated_samples: 生成样本，shape [N, 2]
        gmm: 保留旧接口的参数（未使用）
        path: 保存路径
        title: 图像标题
        n_real_samples: 保留旧接口的参数（未使用）
    """
    if generated_samples.shape[1] < 2:
        return
    
    # 避免未使用参数的linter告警
    _ = (gmm, n_real_samples)

    generated_samples = generated_samples.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1],
               s=12, alpha=0.6, color="tab:blue", label="Generated")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.set_title("Diffusion Generated Samples" + (f" - {title}" if title else ""))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args: Args):
    # Setup DDP:
    device = int(os.environ.get("LOCAL_RANK"))
    rank = dist.get_rank()

    log_dir = f"./local_output/{args.exp_name}"
    # Setup an experiment folder:
    if rank == 0:
        checkpoint_dir = os.path.join(args.base_dir, 'exps', args.project_name, args.exp_name, 'ckpts')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        logger = create_logger(log_dir)
        if args.wandb:
            wandb.init(
                project='toy_dmd',
                resume='auto',
                save_code=True,
                id=args.exp_name,
                name=args.exp_name,
                config=args.state_dict()
            )
            wandb.run.log_code(".", include_fn=lambda x: x.endswith('.py'))
    else:
        checkpoint_dir = None
        logger = create_logger(None)

    # log args
    logger.info(f"Args: {args.state_dict()}")

    # Create GMM using gmm.py
    if args.gmm_dim != 2:
        raise ValueError(f"GMM currently only supports dim=2, got {args.gmm_dim}")
    
    # Set random seed for GMM initialization
    np.random.seed(args.seed)
    gmm = SShapeDistribution2D(random_state=42)
    
    # Create model: use LightningDiT-Mini/1 for GMM fitting
    # Input shape: [B, gmm_dim, 1, 1] where the channel dimension encodes the mixture data
    logger.info(f"Loading LightningDiT model: {args.model} for GMM fitting")
    logger.info(f"GMM config: {args.gmm_num_components} components, dim={args.gmm_dim}")
    logger.info(f"Input shape: [B, {args.gmm_dim}, 1, 1] (C={args.gmm_dim}, H=W=1)")
    
    args.num_classes = args.gmm_num_components  # Use number of components as num_classes
    
    # All spatial information fits in a single 1x1 patch; gmm_dim is embedded via the channel axis.
    model = LightningDiT_models[args.model](
        input_size=1,
        in_channels=args.gmm_dim,
        num_classes=args.num_classes,
    )
    
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    transport_sampler = Sampler(transport)

    logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, fused=True)

    # Prepare model for training
    model.train()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (GMM component indices):
    n_samples = 25
    ys = torch.randint(0, args.gmm_num_components, size=(n_samples,), device=device) * 0
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise for GMM: [B, gmm_dim, 1, 1]
    zs = torch.randn(n_samples, args.gmm_dim, 1, 1, device=device)

    # Setup classifier-free guidance:
    sample_model_kwargs = dict(y=ys)
    model_fn = model.module.forward

    logger.info(f"Training for {args.epochs} epochs...")
    attn_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]) if args.flash_attn else nullcontext()
    
    # Calculate number of batches per epoch (approximate)
    batches_per_epoch =1000
    
    with attn_context:
        for epoch in range(args.epochs):
            logger.info(f"Beginning epoch {epoch}...")
            for cur_iter in range(batches_per_epoch):
                # Generate GMM samples with matching labels using gmm.py
                samples_np, labels_np = gmm.sample(args.local_bs)
                
                # Convert to PyTorch tensors
                x = torch.from_numpy(samples_np).float().to(device)  # [B, gmm_dim]
                y = torch.from_numpy(labels_np).long().to(device)  # [B]
                y = torch.zeros_like(y) # force unconditional training
                
                # Reshape to [B, gmm_dim, 1, 1] so the mixture dimension lives on the channel axis.
                x = x.unsqueeze(-1).unsqueeze(-1)  # [B, gmm_dim, 1, 1]
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_kwargs = dict(y=y)
                    t, loss_dict = transport.training_losses(model, x, model_kwargs) 

                loss = loss_dict["loss"].mean().float()
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

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
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {grad_norm:.4f}")

                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "opt": opt.state_dict(),
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()
                
                if (train_steps % args.sample_every == 0 and train_steps > 0) or train_steps == 1:
                    logger.info("Generating samples for visualization...")
                    model.eval()
                    
                    # Generate more samples for visualization (1000 points)
                    n_viz_samples = 1000
                    zs_viz = torch.randn(n_viz_samples, args.gmm_dim, 1, 1, device=device)
                    ys_viz = torch.zeros(n_viz_samples, dtype=torch.long, device=device)  # unconditional
                    
                    sample_fn = transport_sampler.sample_ode()
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            # sample_fn signature: sample(x, model, **model_kwargs)
                            samples_viz = sample_fn(zs_viz, model_fn, y=ys_viz)[-1]
                    
                    model.train()
                    dist.barrier()

                    # samples shape: [B, gmm_dim, 1, 1] from model output
                    samples_viz = samples_viz.squeeze(-1).squeeze(-1)  # [B, gmm_dim]
                    
                    # Gather samples from all processes
                    all_samples_viz = torch.zeros((n_viz_samples*dist.get_world_size(), args.gmm_dim), device=device)
                    tdist.all_gather_into_tensor(all_samples_viz, samples_viz)
                    
                    if rank == 0:         
                        # Visualize generated samples vs real GMM
                        if args.plot_samples and args.gmm_dim >= 2:
                            plot_path = os.path.join(checkpoint_dir, f"gmm_samples_{train_steps:07d}.png")
                            plot_gmm_samples(
                                all_samples_viz, 
                                gmm, 
                                plot_path, 
                                title=f"Step {train_steps}"
                            )
                            logger.info(f"Saved visualization to {plot_path}")
                    
                    
                    logger.info("Generating samples done.")
                

    logger.info("Done!")


def init_args() -> Args:
    args: Args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:    
        args.exp_name = "debug_" + args.exp_name
    dist.init_distributed_mode(f"./local_output/{args.exp_name}")
    args.device = dist.get_device()
    args.global_bs = args.local_bs * dist.get_world_size()
    args.seed_everything()
    args.set_tf32(args.tf32)
    return args

# torchrun --nproc_per_node=1 train_diffusion.py
if __name__ == "__main__":
    args:Args = init_args()
    main(args)