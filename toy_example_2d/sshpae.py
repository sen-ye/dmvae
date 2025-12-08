import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SShapeDistribution2D:
    def __init__(
        self,
        thickness: float = 0.06,
        diffusion: float = 0.03,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        amplitude: float = 0.85,
        vertical_scale: float = 0.85,
        skew: float = 0.15,
        flip_y: bool = True,
        random_state: Optional[int] = None,
    ):
        self.thickness = thickness
        self.diffusion = diffusion
        self.x_range = x_range
        self.y_range = y_range
        self.amplitude = amplitude
        self.vertical_scale = vertical_scale
        self.skew = skew
        self.flip_y = flip_y
        self.rng = np.random.default_rng(random_state)

    def _skeleton_curve(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """S形主干曲线参数方程"""
        x = self.amplitude * np.sin(np.pi * t)
        # 纵向主干 + 轻微对称扰动，使得上下两端略鼓起
        y = self.vertical_scale * t - self.skew * np.sin(2 * np.pi * t)
        return x, y

    def _perpendicular_angles(self, t: np.ndarray) -> np.ndarray:
        """计算曲线在参数t处的法线角度，用于生成厚度噪声"""
        dx_dt = self.amplitude * np.pi * np.cos(np.pi * t)
        dy_dt = self.vertical_scale - 2 * np.pi * self.skew * np.cos(2 * np.pi * t)
        tangents = np.stack([dx_dt, dy_dt], axis=1)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
        normal_angles = np.arctan2(tangents[:, 0], -tangents[:, 1])  # 90度旋转
        return normal_angles

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样n个点，返回 (samples, component_indices)
        component_indices 全部为0，表示单一组件
        """
        t = self.rng.uniform(-1.0, 1.0, size=n)
        core_x, core_y = self._skeleton_curve(t)
        samples = np.stack([core_x, core_y], axis=1)

        # 法线方向上的局部厚度
        angles = self._perpendicular_angles(t)
        radial_noise = self.rng.normal(0.0, self.thickness, size=n)
        samples[:, 0] += radial_noise * np.cos(angles)
        samples[:, 1] += radial_noise * np.sin(angles)

        # 边缘更为散漫：|t| 越大噪声越强
        diffusion_scale = self.diffusion * (0.4 + 0.6 * np.abs(t))
        samples += self.rng.normal(0.0, diffusion_scale[:, None])

        if self.flip_y:
            samples[:, 1] *= -1

        samples[:, 0] = np.clip(samples[:, 0], *self.x_range)
        samples[:, 1] = np.clip(samples[:, 1], *self.y_range)

        component_indices = np.zeros(n, dtype=int)
        return samples, component_indices

    def visualize(
        self,
        n: int,
        path: str,
        figsize: Tuple[int, int] = (8, 8),
        alpha: float = 0.6,
        s: float = 20,
    ):
        """采样并保存散点图"""
        samples, _ = self.sample(n)

        plt.figure(figsize=figsize)
        plt.scatter(samples[:, 0], samples[:, 1], c="#faad07", s=s, alpha=alpha, linewidths=0)

        plt.xlabel("")
        plt.ylabel("")
        # plt.title(f'S-shape Distribution ({n} samples)')
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.5))
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", labelsize=16)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    s_shape = SShapeDistribution2D(random_state=42, flip_y=True)
    s_shape.visualize(1536, "s_shape.png")