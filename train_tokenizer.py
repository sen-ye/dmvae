from copy import deepcopy
import os, random
import numpy as np
from typing import Union, OrderedDict, Optional
import wandb
import utils.dist as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tap import Tap
from models.vae import VAE
from models.dinodisc import DinoDisc
from models.patchgan import NLayerDiscriminator
from utils.build_dataset import build_dataset
from models.init_param import init_weights
from utils.lpips import LPIPS
import torch.nn.functional as F
from utils.diffaug import DiffAug
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext
from evaluation.fid import FID
from evaluation.metrics import PSNR
from contextlib import nullcontext


class Args(Tap):
    # logging / exp
    project_name: str = "DMVAE"
    exp_name: str = "train"
    log_dir: str = None
    # dataset related
    dataset_type: str = "imagenet"
    dataset_path: str = ""
    workers: int = 4
    hflip: bool = True
    same_seed_for_all_ranks: bool = False
    global_bs: int = 256
    local_bs: int = 32
    image_size: int = 256
    # VAE related
    vae_model_size: str = "large"
    z_channels: int = 32
    freeze_encoder: bool = True
    freeze_mlp: bool = False
    vae_ckpt_path: str = ""
    disc_type: str = "patchgan"
    dino_kernel_size: int = 9
    dino_path: str = f""
    disc_norm: str = "sbn"
    disc_specnorm: bool = False
    conv_std_or_gain: float = 0.02
    disc_init: float = 0.02
    disc_lr: float = 1e-4
    disc_wd: float = 0.0005
    # training related
    seed: int = 42
    tf32: bool = True
    bf16: bool = True
    epochs: int = 1000
    grad_accu: int = 1
    lr: float = 1e-4
    wd: float = 0.005
    ema_decay: float = 0.999
    checkpoint_every: int = 50000
    # device
    device: str = None
    # vae & disc training
    lpips_path: str = f""
    l1: float = 1.0
    l2: float = 0.0
    lpips: float = 1.0
    bcr: float = 1.0
    bcr_cut: float = 0.2
    disc_start_step: int = 5000
    disc_weight: float = 0.5
    flash_attn: bool = True
    eval_every: int = 10000
    lr_warmup_steps: int = 1000

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

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')


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

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:  # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def weight_warmup(step, warmup_steps, max_weight):
    if step < warmup_steps:
        return step / warmup_steps * max_weight
    return max_weight


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


class VAELossFunction:
    def __init__(self, args: Args,
                 vae_wo_ddp: VAE,
                 vae_ddp: VAE,
                 disc_wo_ddp: DinoDisc,
                 disc_ddp: DinoDisc,
                 vae_optimizer: torch.optim.Optimizer,
                 disc_optimizer: torch.optim.Optimizer,
                 ):
        self.args = args
        self.vae_wo_ddp = vae_wo_ddp
        self.vae_ddp = vae_ddp
        self.disc_wo_ddp = disc_wo_ddp
        self.disc_ddp = disc_ddp
        self.vae_optimizer = vae_optimizer
        self.disc_optimizer = disc_optimizer
        self.lpips_loss = LPIPS(ckpt_path=args.lpips_path).eval().requires_grad_(False).cuda()
        self.l1 = args.l1
        self.l2 = args.l2
        self.lpips = args.lpips
        self.disc_weight = args.disc_weight
        self.daug = DiffAug(prob=1.0, cutout=0.2)
        self.bcr_weight = args.bcr
        self.bcr_strong_aug = DiffAug(prob=1, cutout=self.args.bcr_cut)


    def forward_generator(self, images_pm1, recon_image, step=0):
        l1 = F.l1_loss(recon_image, images_pm1)
        l2 = F.mse_loss(recon_image, images_pm1)
        lpips = self.lpips_loss(images_pm1, recon_image).mean()
        rec_loss = l1 * self.l1 + l2 * self.l2 + lpips * self.lpips
        log = {
            "L1": l1.detach().item(),
            "L2": l2.detach().item(),
            "LPIPS": lpips.detach().item(),
            "rec_loss": rec_loss.detach().item(),
        }
        if self.disc_weight > 0 and step >= self.args.disc_start_step:
            self.disc_wo_ddp.eval()
            requires_grad(self.disc_wo_ddp, False)
            d_loss = -self.disc_ddp(self.daug.aug(recon_image, 0), grad_ckpt=False).mean()
            last_layer = self.vae_wo_ddp.decoder.get_last_layer()
            w = (torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0].data.norm() /
                 torch.autograd.grad(d_loss, last_layer, retain_graph=True)[0].data.norm().add_(1e-6))
            w.clamp_(0.0, 1e4)
            d_weight = self.disc_weight * w
            d_loss = d_loss * d_weight
            rec_loss = rec_loss + d_loss
            log.update({
                "d_weight": d_weight.mean().detach().item(),
            })
        return rec_loss, log


    def forward_discriminator(self, images_pm1, recon_image, fade_blur_schedule=0.0):
        requires_grad(self.disc_wo_ddp, True)
        self.disc_wo_ddp.train()
        bs = images_pm1.size(0)
        recon_image_no_grad = torch.cat([images_pm1, recon_image], dim=0)
        logits = self.disc_ddp(self.daug.aug(recon_image_no_grad, fade_blur_schedule), grad_ckpt=False).float()
        logits_real, logits_fake = logits[:bs], logits[bs:]
        d_loss = 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())
        acc_real = (logits_real.data > 0).float().mean().mul_(100)
        acc_fake = (logits_fake.data < 0).float().mean().mul_(100)
        logits2 = self.disc_ddp(self.bcr_strong_aug.aug(recon_image_no_grad, 0.0)).float()
        Lbcr = F.mse_loss(logits2, logits).mul_(self.bcr_weight)
        log = {
            "d_loss": d_loss.mean().detach().item(),
            "bcr_loss": Lbcr.mean().detach().item(),
            "acc_real": acc_real.item(),
            "acc_fake": acc_fake.item(),
            "acc_mean": ((acc_real + acc_fake) * 0.5).item(),
        }
        d_loss = d_loss + Lbcr
        return d_loss, log
    

def init_args() -> Args:
    args: Args = Args(explicit_bool=True).parse_args(known_only=True)
    dist.init_distributed_mode(f"./local_output/{args.exp_name}")
    args.device = dist.get_device()
    args.seed_everything()
    args.set_tf32(args.tf32)
    args.global_bs = args.local_bs * dist.get_world_size()

    args.log_dir = f"{args.log_dir}/{args.project_name}/{args.exp_name}/ckpts/"

    if dist.is_master():
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.init(
            project=args.project_name,
            resume='auto',
            save_code=True,
            id=args.exp_name,
            name=args.exp_name,
            config=args.state_dict()
        )
        wandb.run.log_code(".", include_fn=lambda x: x.endswith('.py'))
        print(f"Args: {args.state_dict()}")
    return args

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

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def build_models(args: Args):
    local_rank = dist.get_local_rank()
    vae = VAE(
        z_channels=args.z_channels,
        image_size=args.image_size,
        model_size=args.vae_model_size,
        conv_std_or_gain=0.02, 
    ).to(args.device)
    if args.vae_ckpt_path:
        vae.load_state_dict(torch.load(args.vae_ckpt_path, weights_only=False, map_location="cpu"))
        print(f"[VAE] loaded checkpoint from {args.vae_ckpt_path}")
    if args.freeze_encoder:
        vae.encoder.eval()
        requires_grad(vae.encoder, False)
    if args.freeze_mlp:
        vae.bottle_neck.eval()
        requires_grad(vae.bottle_neck, False)   

    vae_ddp = DDP(vae, device_ids=[local_rank], find_unused_parameters=False)

    # Discriminator
    disc_ddp, disc_wo_ddp = None, None
    if args.disc_weight > 0:
        if args.disc_type == 'dino':
            disc = DinoDisc(
                device=args.device,
                ks=args.dino_kernel_size,
                dino_ckpt=args.dino_path,
                key_depths=(0, 2, 5, 8, 11),
                norm_type=args.disc_norm, norm_eps=1e-6, use_specnorm=args.disc_specnorm
            ).to(args.device)
        else:
            disc = NLayerDiscriminator().to(args.device)
            init_weights(disc, 0.02)
        disc_wo_ddp = disc
        disc_ddp = DDP(disc, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    return vae_ddp, vae, disc_ddp, disc_wo_ddp


@torch.inference_mode()
def eval(
    args: Args,
    vae_wo_ddp: VAE,
    eval_data,
    ema=False,
):
    vae_wo_ddp.eval()
    device = args.device
    psnr = 0

    compute_fid_score = FID().to(device)

    dataloader = eval_data.dataloader
    num_sample = eval_data.num_samples
    latent_mean = torch.tensor(0, device=device, dtype=torch.float32)
    latent_scale = torch.tensor(0, device=device, dtype=torch.float32)
    total_batch = torch.tensor(0, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for sample in dataloader:
            sample, _ = sample
            sample = sample.to(device, non_blocking=True)
            latent = vae_wo_ddp.encode(sample)
            x_rec = vae_wo_ddp.decode(latent)
            latent_mean += latent.float().detach().mean()
            latent_scale += 1 / (latent.float().detach().std() + 1e-8)
            total_batch += 1
            x_rec = (x_rec + 1) / 2
            sample = (sample + 1) / 2
            psnr += PSNR(x_rec, sample, reduce="sum")
            compute_fid_score.update(x_rec, False)
            compute_fid_score.update(sample, True)

    dist.allreduce(psnr), dist.allreduce(latent_mean), dist.allreduce(latent_scale), dist.allreduce(total_batch)
    dist.barrier()
    psnr = psnr / num_sample
    latent_mean = latent_mean / total_batch
    latent_scale = latent_scale / total_batch
    fid = compute_fid_score.compute()
    ema_string = "_ema" if ema else ""
    print(f'[eval{ema_string}] PSNR: {psnr:.3f}, FID: {fid:.3f}, latent_mean: {latent_mean:.3f}, latent_scale: {latent_scale:.3f}')
    wandb_log({f'PSNR{ema_string}': psnr, f'rFID{ema_string}': fid, f'latent_mean{ema_string}': latent_mean.item(), f'latent_scale{ema_string}': latent_scale.item()})
    dist.barrier()
    vae_wo_ddp.train()
 
    

def main(args: Args):
    vae_ddp, vae_wo_ddp, disc_ddp, disc_wo_ddp = build_models(args)
    train_data = build_dataset(mode="train", dataset_path=args.dataset_path,
            workers=args.workers, same_seed_for_all_ranks=args.same_seed_for_all_ranks, global_bs=args.global_bs, local_bs=args.local_bs, get_different_generator_for_each_rank=args.get_different_generator_for_each_rank)
    eval_data = build_dataset(mode="val", dataset_path=args.dataset_path,
            workers=0, same_seed_for_all_ranks=args.same_seed_for_all_ranks, global_bs=args.global_bs, local_bs=args.local_bs, get_different_generator_for_each_rank=args.get_different_generator_for_each_rank)
    dataloader = train_data.dataloader
    datasampler = train_data.sampler
    num_iters_per_epoch = train_data.num_batches

    vae_params_train = [p for p in vae_wo_ddp.parameters() if p.requires_grad]
    optimizer_vae = torch.optim.AdamW(vae_params_train, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95), eps=1e-8)
    optimizer_disc = torch.optim.AdamW(disc_wo_ddp.parameters(), lr=args.disc_lr, weight_decay=args.disc_wd, betas=(0.9, 0.95), eps=1e-8) if args.disc_weight > 0 else None

    warmup_steps = args.lr_warmup_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler_vae = torch.optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda)
    scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda) if optimizer_disc else None

    global_step = 0  
    vae_loss_fn = VAELossFunction(args, vae_wo_ddp, vae_ddp, disc_wo_ddp, disc_ddp, optimizer_vae, optimizer_disc,)
    
    ema_model = deepcopy(vae_wo_ddp)
    requires_grad(ema_model, False)
    ema_model.eval()

    attn_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]) if args.flash_attn else nullcontext()
    with attn_context:
        for epoch in range(args.epochs):
            train_data.set_epoch(epoch)
            for iter in range(num_iters_per_epoch):
                images_pm1, labels = next(dataloader)
                images_pm1 = images_pm1.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recon_image = vae_ddp(images_pm1, return_latent=False)
                    vae_loss, vae_loss_dict = vae_loss_fn.forward_generator(images_pm1, recon_image, step=global_step)
                    vae_loss = vae_loss.mean()
                    vae_loss.backward()
                    g_norm = torch.nn.utils.clip_grad_norm_(vae_ddp.parameters(), max_norm=1.0)
                    vae_loss_dict["vae_norm"] = g_norm.item()
                    optimizer_vae.step()
                    optimizer_vae.zero_grad(set_to_none=True)
                    scheduler_vae.step()
                    if args.disc_weight > 0 and global_step >= args.disc_start_step:
                        d_loss, d_log = vae_loss_fn.forward_discriminator(images_pm1, recon_image.detach())
                        d_loss = d_loss.mean()
                        d_loss.backward()
                        d_norm = torch.nn.utils.clip_grad_norm_(disc_ddp.parameters(), max_norm=1.0)
                        d_log["disc_norm"] = d_norm.item()
                        optimizer_disc.step()
                        optimizer_disc.zero_grad(set_to_none=True)
                        scheduler_disc.step()
                    wandb_log(vae_loss_dict, log_ferq=25, step=global_step)
                    if global_step % 25 == 0:
                        print(f"[Epoch {epoch}] [Iter {iter}] vae_loss_dict: {vae_loss_dict};")
                    if args.disc_weight > 0 and global_step >= args.disc_start_step:
                        wandb_log(d_log, log_ferq=25, step=global_step)
                        if global_step % 25 == 0:
                            print(f"[Epoch {epoch}] [Iter {iter}] d_log: {d_log};")

                update_ema(ema_model, vae_wo_ddp)
                # save checkpoint
                if dist.is_master() and (global_step % args.checkpoint_every == 0):
                    ckpt = {
                        "vae_wo_ddp": vae_wo_ddp.state_dict(),
                        "vae_ema": ema_model.state_dict(),
                        "disc_wo_ddp": disc_wo_ddp.state_dict(),
                        "opt_vae": optimizer_vae.state_dict() if optimizer_vae is not None else None,
                        "opt_disc": optimizer_disc.state_dict() if optimizer_disc is not None else None,
                        "scheduler_vae": scheduler_vae.state_dict(),
                        "scheduler_disc": scheduler_disc.state_dict() if scheduler_disc is not None else None,
                        "args": args.state_dict(),
                        "steps": global_step,
                    }
                    save_path = os.path.join(args.log_dir, f"{global_step:07d}.pt")
                    torch.save(ckpt, save_path)
                    print(f"Saved checkpoint to {save_path}")

                # eval
                if (global_step % args.eval_every == 0):
                    eval(args, vae_wo_ddp, eval_data, ema=False)
                    eval(args, ema_model, eval_data, ema=True)
                
                global_step += 1

if __name__ == "__main__":
    args = init_args()
    main(args)