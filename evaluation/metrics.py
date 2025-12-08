import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure # type: ignore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # type: ignore


def PSNR(img1, img2, reduce="sum"):
    # input shape: [B, C, H, W] scaled to [0, 1]
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse)
    if reduce == "sum":
        return torch.sum(psnr)
    elif reduce == "mean":
        return torch.mean(psnr)


def SSIM(pred, gt, data_range=1.0):
    # input shape: [T, C, H, W]
    pred = pred.add(1).mul(0.5)
    gt = gt.add(1).mul(0.5)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction='none').to(pred.device)
    ssim_t = ssim(pred, gt)
    ssim_mean = torch.mean(ssim_t)
    return ssim_t, ssim_mean


def LPIPS(img1, img2, net_type='alex'):
    # input shape: [T, C, H, W] scaled to [-1, 1]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction='mean').to(img1.device)
    return lpips(img1, img2)
    