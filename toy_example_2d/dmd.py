from copy import deepcopy
import os, random, math
import numpy as np
from typing import Union, OrderedDict, Optional
from collections import defaultdict
import wandb
import utils.dist as dist
import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from tap import Tap
import torch.nn as nn
import torch.nn.functional as F
from diffusion.lightningdit.lightningdit import LightningDiT_models
from diffusion.transport import create_transport, Sampler
from diffusion.lightningdit.lightningdit import LightningDiT
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sshpae import SShapeDistribution2D


def wandb_log(log_dict, step=None, log_ferq=1):
    """简单的wandb日志记录函数"""
    if wandb.run is not None and step is not None and step % log_ferq == 0:
        wandb.log(log_dict, step=step)
    elif wandb.run is not None and step is None:
        wandb.log(log_dict)


class DMDArgs(Tap):
    # logging / exp
    project_name: str = "toy_dmd"
    exp_name: str = "train_dmd"
    log_dir: str = None
    # points related
    num_points: int = 1536  # 点的数量
    z_channels: int = 2  # 点的维度（x, y）
    # DiT related
    fake_model: str = "LightningDiT-Mini/1"
    fake_model_ckpt_path: str = ""
    fake_model_ckpt_ema: bool = False
    real_model: str = "LightningDiT-Mini/1"
    real_model_ckpt_path: str = ""
    real_model_ckpt_ema: bool = False
    num_classes: int = 1  # 类别数量（简化，使用单一类别）
    # training related
    seed: int = 42
    tf32: bool = True
    bf16: bool = True
    epochs: int = 10
    grad_accu: int = 1
    lr: float = 1e-4
    diff_lr: float = 1e-4
    wd: float = 0.0
    ema_decay: float = 0.999
    checkpoint_every: int = 50000
    # sampling cfg
    cfg_scale: float = 4.0
    ode_num_steps: int = 25
    # transport config
    path_type: str = "Linear"
    prediction: str = "velocity"
    train_eps: float = 0.0
    sample_eps: float = 0.0
    # device
    device: str = None
    # dmd training
    dmd_weight: float = 1.0
    flash_attn: bool = True
    eval_every: int = 10000
    t1: float = 1.0
    t0: float = 0.0
    resume_from: str = ""
    reinit_fake: bool = False
    fake_warmup_steps: int = 0
    # loss type: "dmd" (original)
    dmd_loss_type: str = "dmd"
    # training frequency
    vae_train_every: int = 5  # 每N步训练一次points

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:
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
            if k not in {'device', 'dbg_ks_fp'}:
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def create_learnable_points(num_points: int, z_channels: int, device: str, seed: int = 42):
    """创建可学习的点，初始化为[-1.5, 1.5]均匀分布"""
    torch.manual_seed(seed)
    # 在[-1.5, 1.5]范围内均匀分布
    points = torch.rand(num_points, z_channels, device=device) * 3.0 - 1.5  # [num_points, z_channels]
    points = nn.Parameter(points)
    return points

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
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x"""
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


POINT_COLOR = "#faad07"
ARROW_COLOR = "#75726b"
TRIANGLE_POINT_COLOR = "#67B2D8"
SSHAPE_SAMPLE_COUNT = 1536
TRIANGLE_POINTS_LEFT = np.array([
    [-1.0, -1.0],
    [-0.75, -1.0],
    [-0.5, -1.0],
    [-1.0, -0.75],
    [-1.0, -0.5],
    [-0.75, -0.75],
], dtype=np.float32)


def _style_axis_common(ax, axis_limit: float):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)


def _scatter_points(ax, points_2d: np.ndarray, axis_limit: float, s: float = 20, alpha: float = 0.6):
    ax.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        alpha=alpha,
        s=s,
        c=POINT_COLOR,
        edgecolors='none',
    )
    _style_axis_common(ax, axis_limit)


def _plot_gradient_field(
    ax,
    base_points: np.ndarray,
    gradients: np.ndarray,
    axis_limit: float,
    point_size: float = 30,
    point_color: str = POINT_COLOR,
    arrow_width: float = 0.003,
):
    ax.scatter(
        base_points[:, 0],
        base_points[:, 1],
        alpha=0.8,
        s=point_size,
        c=point_color,
        edgecolors='white',
        linewidths=0.3,
    )
    grad_vec = gradients.copy()
    grad_magnitude = np.linalg.norm(grad_vec, axis=1, keepdims=True)
    grad_magnitude = np.clip(grad_magnitude, 1e-8, None)
    grad_direction = grad_vec / grad_magnitude
    max_magnitude = grad_magnitude.max()
    normalized = grad_magnitude / max_magnitude if max_magnitude > 0 else np.ones_like(grad_magnitude)
    arrow_scale = axis_limit * 0.25
    arrows = grad_direction * normalized * arrow_scale
    ax.quiver(
        base_points[:, 0],
        base_points[:, 1],
        arrows[:, 0],
        arrows[:, 1],
        angles='xy',
        scale_units='xy',
        scale=1.0,
        width=arrow_width,
        color=ARROW_COLOR,
        alpha=0.8,
    )
    _style_axis_common(ax, axis_limit)


def _corner_triangle_points(axis_limit: float) -> np.ndarray:
    """
    构建左下角与右上角的固定三角形点阵，点坐标基于axis_limit比例。
    """
    scale = axis_limit / 1.5
    left = TRIANGLE_POINTS_LEFT * scale
    right = -left  # 中心对称
    return np.concatenate([left, right], axis=0)


def _save_figure(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_avg_gradients(
    step: int,
    axis_limit: float,
    points_2d: Optional[np.ndarray],
    grads_all_t: Optional[list],
    grads_high_t: Optional[list],
    save_dir: str,
):
    """
    保存全t平均梯度与高t平均梯度的箭头图，并返回需记录到WandB的字典。
    """
    log_payload = {}
    if points_2d is None:
        return log_payload
    os.makedirs(save_dir, exist_ok=True)
    
    if grads_all_t:
        stacked = np.stack(grads_all_t, axis=0)
        avg_grad = stacked.mean(axis=0)
        avg_path = os.path.join(save_dir, f'eval_step{step}_avg_all_t_grad.png')
        fig_avg, ax_avg = plt.subplots(figsize=(8, 8))
        _plot_gradient_field(ax_avg, points_2d, avg_grad, axis_limit)
        _save_figure(fig_avg, avg_path)
        print(f'[eval] Saved average gradient visualization to {avg_path}')
        log_payload['eval_avg_grad_all_t'] = wandb.Image(avg_path)
    
    if grads_high_t:
        stacked_high = np.stack(grads_high_t, axis=0)
        high_grad = stacked_high.mean(axis=0)
        high_path = os.path.join(save_dir, f'eval_step{step}_avg_t0.7_0.9_grad.png')
        fig_high, ax_high = plt.subplots(figsize=(8, 8))
        _plot_gradient_field(ax_high, points_2d, high_grad, axis_limit)
        _save_figure(fig_high, high_path)
        print(f'[eval] Saved high-t gradient visualization to {high_path}')
        log_payload['eval_avg_grad_t0.7_0.9'] = wandb.Image(high_path)
    
    return log_payload




class DMDLossFunction:
    def __init__(self, args: DMDArgs,
                 sit_wo_ddp: LightningDiT,
                 sit_ddp: LightningDiT,
                 base_model: LightningDiT,
                 transport,
                 ):
        self.args = args
        self.dmd_weight = args.dmd_weight
        self.transport = transport
        self.sit_wo_ddp = sit_wo_ddp
        self.sit_ddp = sit_ddp
        self.base_model = base_model

    def compute_distribution_matching_loss(self, points: torch.Tensor, labels: torch.Tensor, step: int = 0, force_t=None):
        """计算DMD损失
        Args:
            points: [B, z_channels] 可学习的点
            labels: [B] 类别标签
        Returns:
            loss: 损失值
            log: 日志字典
            grad: 梯度（用于更新点）
        """
        # points: [B, z_channels] -> 转换为空间格式 [B, z_channels, 1, 1]
        if points.dim() == 2:
            points_spatial = points.unsqueeze(-1).unsqueeze(-1)  # [B, z_channels, 1, 1]
        else:
            points_spatial = points
    
        t, x0, x1 = self.transport.sample(points_spatial)
        t0 = self.args.t0
        t1 = self.args.t1
        if force_t is None:
            t = t * (t1 - t0) + t0
        else:
            t = torch.ones_like(t) * force_t
        t = t.to(points_spatial.device)
        _, xt, _ = self.transport.path_sampler.plan(t, x0, points_spatial)
        
        # 根据loss类型计算loss和grad
        loss_type = self.args.dmd_loss_type
        
        if loss_type == "dmd":
            # 原始的DMD loss
            with torch.no_grad():
                v_teacher = self.base_model(xt, t, labels)
                v_student = self.sit_wo_ddp(xt, t, labels)
            pred_teacher = xt + v_teacher * expand_t_like_x(1 - t, xt)
            pred_student = xt + v_student * expand_t_like_x(1 - t, xt)
            p_real = (points_spatial - pred_teacher)
            p_student = (points_spatial - pred_student)
            grad = (p_real - p_student)
            grad = torch.nan_to_num(grad)
            loss = 0.5 * torch.nn.functional.mse_loss(points_spatial, (points_spatial-grad).detach(), reduction="mean")
        elif loss_type == "score_teacher":
            with torch.no_grad():
                v_teacher = self.base_model(xt.detach(), t, labels)
            # 从velocity计算score，使用path_sampler.get_score_from_velocity
            score_teacher = self.transport.path_sampler.get_score_from_velocity(v_teacher, xt, t)
            grad = score_teacher
            grad = torch.nan_to_num(grad)
            # loss是score_teacher的L2范数的平方（用于监控）
            loss = 0.5 * torch.nn.functional.mse_loss(points_spatial, (points_spatial+grad).detach(), reduction="mean")
        elif loss_type == "score_student":
            with torch.no_grad():
                v_student = self.sit_wo_ddp(xt, t, labels)
            score_student = self.transport.path_sampler.get_score_from_velocity(v_student, xt, t)
            grad = score_student
            grad = torch.nan_to_num(grad)
            loss = 0.5 * torch.nn.functional.mse_loss(points_spatial, (points_spatial+grad).detach(), reduction="mean")
        elif loss_type == "score_student_minus":
            with torch.no_grad():
                v_student = self.sit_wo_ddp(xt, t, labels)
            score_student = self.transport.path_sampler.get_score_from_velocity(v_student, xt, t)
            grad = score_student
            grad = torch.nan_to_num(grad)
            loss = 0.5 * torch.nn.functional.mse_loss(points_spatial, (points_spatial-grad).detach(), reduction="mean")
        elif loss_type == "loss_teacher1":
            v_teacher = self.base_model(xt, t, labels)
            loss = 0.5 * torch.nn.functional.mse_loss(v_teacher, (x1 - x0).detach(), reduction="mean")
        elif loss_type == "loss_teacher2":
            v_teacher = self.base_model(xt.detach(), t, labels)
            loss = 0.5 * torch.nn.functional.mse_loss((x1 - x0), v_teacher.detach(), reduction="mean")
        elif loss_type == "loss_teacher3":
            v_teacher = self.base_model(xt, t, labels)
            loss = 0.5 * ((v_teacher - (x1 - x0))**2).mean()
        elif loss_type == "loss_student1":
            v_student = self.sit_wo_ddp(xt, t, labels)
            loss = 0.5 * torch.nn.functional.mse_loss(v_student, (x1 - x0).detach(), reduction="mean")
        elif loss_type == "loss_student2":
            v_student = self.sit_wo_ddp(xt.detach(), t, labels)
            loss = 0.5 * torch.nn.functional.mse_loss((x1 - x0), v_student.detach(), reduction="mean")
        elif loss_type == "loss_student3":
            v_student = self.sit_wo_ddp(xt, t, labels)
            loss = 0.5 * ((v_student - (x1 - x0))**2).mean()
        elif loss_type == "score_teacher_student1":
            v_teacher = self.base_model(xt.detach(), t, labels).detach()
            v_student = self.sit_wo_ddp(xt, t, labels)
            loss = ((v_teacher - v_student)**2).mean()
        elif loss_type == "score_teacher_student2":
            v_teacher = self.base_model(xt, t, labels)
            v_student = self.sit_wo_ddp(xt.detach(), t, labels).detach()
            loss = ((v_teacher - v_student)**2).mean()
        elif loss_type == "score_teacher_student3":
            v_teacher = self.base_model(xt, t, labels)
            v_student = self.sit_wo_ddp(xt, t, labels)
            loss = ((v_teacher - v_student)**2).mean()
        elif loss_type == "loss_teacher_student1":
            v_teacher = self.base_model(xt, t, labels)
            loss_teacher = 0.5 * torch.nn.functional.mse_loss(v_teacher, (x1 - x0).detach(), reduction="mean")
            v_student = self.sit_wo_ddp(xt, t, labels)
            loss_student = 0.5 * torch.nn.functional.mse_loss(v_student, (x1 - x0).detach(), reduction="mean")
            loss = loss_teacher - loss_student
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        log = {
            "dmd_loss": loss.detach().item(),
            "loss_type": loss_type,
        }
        
        return loss, log


def build_models(args: DMDArgs):
    """构建模型"""
    local_rank = dist.get_local_rank()
    
    # DiT fake model
    sit = LightningDiT_models[args.fake_model](
        input_size=1,
        in_channels=args.z_channels,
        num_classes=args.num_classes
    ).to(args.device)
    if args.fake_model_ckpt_path and not args.reinit_fake:
        try:
            m = torch.load(args.fake_model_ckpt_path, map_location='cpu')
            sit.load_state_dict(m['ema'] if args.fake_model_ckpt_ema else m.get('model', m), strict=True)
            print(f"[Fake model] loaded checkpoint from {args.fake_model_ckpt_path}")
        except Exception as e:
            print(f"[Fake model] load_state_dict failed: {e}.")

    # DiT real model
    base_model = LightningDiT_models[args.real_model](
        input_size=1,
        in_channels=args.z_channels,
        num_classes=args.num_classes
    ).to(args.device)
    base_model.eval()
    requires_grad(base_model, False)
    if args.real_model_ckpt_path:
        try:
            m = torch.load(args.real_model_ckpt_path, map_location='cpu')
            base_model.load_state_dict(m['ema'] if args.real_model_ckpt_ema else m.get('model', m), strict=True)
            print(f"[Real model] loaded checkpoint from {args.real_model_ckpt_path}")
        except Exception as e:
            print(f"[Real model] load_state_dict failed: {e}.")

    sit_ddp = DDP(sit, device_ids=[local_rank], find_unused_parameters=True)
    requires_grad(sit, False)
    sit.eval()

    return sit_ddp, sit, base_model


def eval(
    args: DMDArgs,
    points: torch.nn.Parameter,
    dmd_loss_fn: 'DMDLossFunction',
    step=0,
    force_t=None,
    n_samples_for_viz=100,
    axis_limit: float = 1.5,
):
    """
    评估函数：可视化点的分布和DMD loss的梯度
    
    Args:
        points: 可学习的点 [num_points, z_channels]
        dmd_loss_fn: DMDLossFunction 对象
        step: 当前训练步数
        force_t: 强制使用的时间步 t（如果为 None，则使用随机 t）
        n_samples_for_viz: 用于可视化的样本数量（固定100个）
        axis_limit: 可视化时x/y轴的范围 [-axis_limit, axis_limit]
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (selected_points_2d, grad_2d)
    """
    device = points.device
    num_points = points.shape[0]
    
    # 设置随机种子，确保每次eval选择相同的100个点
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 固定选择前100个点用于可视化
    n_samples = min(n_samples_for_viz, num_points)
    eval_indices = np.arange(n_samples)  # 固定选择前100个点
    
    selected_points = points[eval_indices].detach()  # [n_samples, z_channels]
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)  # 使用单一类别
    
    # 计算梯度
    selected_points_for_grad = selected_points.clone().requires_grad_(True)
    dmd_loss, _ = dmd_loss_fn.compute_distribution_matching_loss(
        selected_points_for_grad, labels, step=step, force_t=force_t
    )
    grad_points = torch.autograd.grad(dmd_loss, selected_points_for_grad, retain_graph=False, create_graph=False)[0]
    
    # 转换为numpy
    points_np = selected_points.float().cpu().numpy()  # [n_samples, z_channels]
    grad_np = - grad_points.float().cpu().numpy()  # [n_samples, z_channels]
    
    # 可视化
    if dist.is_master():
        all_points_np = points.detach().float().cpu().numpy()
        
        # 固定使用前两维（x, y）
        points_2d = points_np[:, :2]
        grad_2d = grad_np[:, :2]
        all_points_2d = all_points_np[:, :2]
        # 保存目录与文件名前缀
        save_dir = f"{args.log_dir}/{args.project_name}/{args.exp_name}/eval/"
        os.makedirs(save_dir, exist_ok=True)
        t_str = f't{force_t}' if force_t is not None else 't_random'
        
        # 1) 所有点分布
        fig_all, ax_all = plt.subplots(figsize=(8, 8))
        _scatter_points(ax_all, all_points_2d, axis_limit, s=20, alpha=0.6)
        all_path = os.path.join(save_dir, f'eval_step{step}_{t_str}_all_points.png')
        _save_figure(fig_all, all_path)
        print(f'[eval] Saved visualization to {all_path}')
        
        # 2) 前100个点梯度场
        fig_grad, ax_grad = plt.subplots(figsize=(8, 8))
        _plot_gradient_field(ax_grad, points_2d, grad_2d, axis_limit)
        grad_path = os.path.join(save_dir, f'eval_step{step}_{t_str}_selected_grad.png')
        _save_figure(fig_grad, grad_path)
        print(f'[eval] Saved visualization to {grad_path}')
        
        # 3) 角点梯度场（左下/右上三角形）
        grid_points_2d = _corner_triangle_points(axis_limit)
        grid_points_full = torch.zeros(
            (grid_points_2d.shape[0], points.shape[1]),
            device=device,
            dtype=points.dtype,
        )
        grid_points_full[:, :2] = torch.from_numpy(grid_points_2d).to(device=device, dtype=points.dtype)
        grid_points_for_grad = grid_points_full.clone().requires_grad_(True)
        labels_grid = torch.zeros(grid_points_full.shape[0], dtype=torch.long, device=device)
        dmd_loss_grid, _ = dmd_loss_fn.compute_distribution_matching_loss(
            grid_points_for_grad,
            labels_grid,
            step=step,
            force_t=force_t,
        )
        grad_grid = torch.autograd.grad(
            dmd_loss_grid,
            grid_points_for_grad,
            retain_graph=False,
            create_graph=False,
        )[0]
        grad_grid_np = -grad_grid.float().cpu().numpy()[:, :2]
        
        s_shape = SShapeDistribution2D(random_state=42)
        s_samples, _ = s_shape.sample(SSHAPE_SAMPLE_COUNT)
        fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
        _scatter_points(ax_grid, s_samples[:, :2], axis_limit, s=20, alpha=0.6)
        _plot_gradient_field(
            ax_grid,
            grid_points_2d,
            grad_grid_np,
            axis_limit,
            point_size=80,
            point_color=TRIANGLE_POINT_COLOR,
            arrow_width=0.0045,
        )
        grid_path = os.path.join(save_dir, f'eval_step{step}_{t_str}_triangle_grad.png')
        _save_figure(fig_grid, grid_path)
        print(f'[eval] Saved visualization to {grid_path}')
        
        wandb_log({
            f'eval_all_{t_str}': wandb.Image(all_path),
            f'eval_selected_grad_{t_str}': wandb.Image(grad_path),
            f'eval_triangle_grad_{t_str}': wandb.Image(grid_path),
        })
    
    dist.barrier()
    
    return points_2d, grad_2d


def init_args() -> DMDArgs:
    args: DMDArgs = DMDArgs(explicit_bool=True).parse_args(known_only=True)
    dist.init_distributed_mode(f"./local_output/{args.exp_name}")
    args.device = dist.get_device()
    args.seed_everything()
    args.set_tf32(args.tf32)

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
        print(f"Args: {args.state_dict()}")

    return args


def main(args: DMDArgs):
    sit_ddp, sit_wo_ddp, base_model = build_models(args)
    
    # 创建1000个可学习的点
    points = create_learnable_points(args.num_points, args.z_channels, args.device, args.seed)
    
    # 创建优化器（只用于sit，points手动更新）
    optimizer_points = torch.optim.AdamW([points], lr=args.lr, weight_decay=0, betas=(0.9, 0.95), eps=1e-8)
    optimizer_sit = torch.optim.AdamW(sit_wo_ddp.parameters(), lr=args.diff_lr, weight_decay=args.wd, betas=(0.9, 0.95), eps=1e-8)

    global_step = 0

    transport = create_transport(
        args.path_type,
        args.prediction,
        None,
        args.train_eps,
        args.sample_eps
    )
    dmd_loss_fn = DMDLossFunction(args, sit_wo_ddp, sit_ddp, base_model, transport)

    scaler_dtype = torch.bfloat16 if args.bf16 else torch.float32

    attn_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]) if args.flash_attn else nullcontext()
    
    # 设置batch size，每次训练使用所有点或一个batch的点
    batch_size = min(args.num_points, 512)  # 每次使用最多256个点
    num_iters_per_epoch = 1000
    
    with attn_context:
        original_vae_train_every = args.vae_train_every
        for epoch in range(args.epochs):
            for iter in range(num_iters_per_epoch):
                # 根据 fake_warmup_steps 调整 vae_train_every
                if global_step < args.fake_warmup_steps:
                    args.vae_train_every = args.fake_warmup_steps
                else:
                    args.vae_train_every = original_vae_train_every
                
                points_training_turn = (global_step % args.vae_train_every == 0)
                points_trained_this_step = False
                
                # 随机选择一批点进行训练
                batch_points = points  # [batch_size, z_channels]
                labels = torch.zeros(batch_points.shape[0], dtype=torch.long, device=args.device)  # 使用单一类别
                
                # 训练 points (DMD loss) - 根据 vae_train_every 控制频率
                if points_training_turn and not points_trained_this_step:
                    points_trained_this_step = True
                    requires_grad(sit_wo_ddp, False)
                    sit_wo_ddp.eval()
                    
                    # 确保points需要梯度
                    points.requires_grad_(True)
                    
                    with torch.autocast(device_type="cuda", dtype=scaler_dtype):
                        dmd_loss, dmd_log = dmd_loss_fn.compute_distribution_matching_loss(
                            batch_points, labels, step=global_step
                        )
                        dmd_loss = dmd_loss.mean()
                        dmd_loss.backward()
                        g_norm = torch.nn.utils.clip_grad_norm_(points, max_norm=100000.0)
                        optimizer_points.step()
                        optimizer_points.zero_grad(set_to_none=True)
                        dmd_log.update({
                            "points_grad_norm": g_norm.item(),
                        })
                    
                    wandb_log(dmd_log, log_ferq=25, step=global_step)
                    if global_step % 25 == 0:
                        print(f"[Epoch {epoch}] [Iter {iter}] dmd_loss: {dmd_loss.item():.6f}, points_grad_norm: {g_norm:.6f}")
                
                # 训练 sit (diffusion loss) - 每次都训练
                requires_grad(sit_wo_ddp, True)
                sit_wo_ddp.train()
                
                with torch.autocast(device_type="cuda", dtype=scaler_dtype):
                    # 将 points 转换为 spatial 格式用于 diffusion training
                    batch_points_spatial = batch_points.unsqueeze(-1).unsqueeze(-1)  # [batch_size, z_channels, 1, 1]
                    
                    # 使用 transport.training_losses 计算 diffusion loss
                    t, loss_dict = transport.training_losses(sit_ddp, batch_points_spatial.detach(), dict(y=labels))
                    diff_loss = loss_dict["loss"].mean()
                    
                    diff_loss.backward()
                    sit_norm = torch.nn.utils.clip_grad_norm_(sit_ddp.parameters(), max_norm=1.0)
                    
                    diff_log = {
                        "sit_loss": diff_loss.detach().item(),
                        "sit_grad_norm": sit_norm.item(),
                    }
                    
                    optimizer_sit.step()
                    optimizer_sit.zero_grad(set_to_none=True)
                
                wandb_log(diff_log, log_ferq=25, step=global_step)
            

                # save checkpoint
                if dist.is_master() and (global_step % args.checkpoint_every == 0):
                    ckpt = {
                        "model": sit_wo_ddp.state_dict(),
                        "points": points.data.cpu(),
                        "opt_sit": optimizer_sit.state_dict(),
                        "args": args.state_dict(),
                        "steps": global_step,
                    }
                    save_path = os.path.join(args.log_dir, f"{global_step:07d}.pt")
                    torch.save(ckpt, save_path)
                    print(f"Saved checkpoint to {save_path}")

                # eval
                ts = [0.1, 0.3, 0.5, 0.7, 0.9]
                if (global_step % args.eval_every == 0):
                    axis_limit_eval = 1.5
                    avg_grad_storage = [] if dist.is_master() else None
                    high_grad_storage = [] if dist.is_master() else None
                    cached_points_2d = None
                    for t in ts:
                        points_eval_2d, grad_eval_2d = eval(
                            args,
                            points,
                            dmd_loss_fn=dmd_loss_fn,
                            step=global_step,
                            force_t=t,
                            n_samples_for_viz=100,
                            axis_limit=axis_limit_eval,
                        )
                        if dist.is_master():
                            if cached_points_2d is None:
                                cached_points_2d = points_eval_2d
                            avg_grad_storage.append(grad_eval_2d)
                            if t in (0.7, 0.9):
                                high_grad_storage.append(grad_eval_2d)
                    if dist.is_master() and cached_points_2d is not None:
                        save_dir = f"{args.log_dir}/{args.project_name}/{args.exp_name}/eval/"
                        log_payload = _plot_avg_gradients(
                            step=global_step,
                            axis_limit=axis_limit_eval,
                            points_2d=cached_points_2d,
                            grads_all_t=avg_grad_storage,
                            grads_high_t=high_grad_storage,
                            save_dir=save_dir,
                        )
                        if log_payload:
                            wandb_log(log_payload)
                
                global_step += 1


if __name__ == "__main__":
    args = init_args()
    main(args)

