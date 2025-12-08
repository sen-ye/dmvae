import os
import torch
from torch import nn
from models.init_param import init_weights
from models.flux_ae import *
from contextlib import nullcontext
from timm.models import create_model


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std


class Denormalize(nn.Module):
    def __init__(self, mean, std,):
        super(Denormalize, self).__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return x * self.std + self.mean


class DINOEncoder(nn.Module):
    def __init__(self,
                 model_size="base",
                 patch_size=16,   
                 image_size=256,               
                 ):
        super().__init__()
        self.dim = {
            'base': 768,
            'large': 1024,
        }[model_size]
        self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model_size == 'base':
            self.model = create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, patch_size=patch_size, img_size=image_size)
        elif model_size == "large":
            self.model = create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, patch_size=patch_size, img_size=image_size)

    def forward(self, x):
        return self.model.forward_features(self.scale(self.de_scale(x)))[:, self.model.num_prefix_tokens:]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)

    def get_last_layer(self):
        return self.mlp[-1].weight


class VAE(nn.Module):
    def __init__(
        self, 
        z_channels: int = 16,
        image_size: int = 256, 
        model_size: str = 'base',
        patch_size: int = 16,
        conv_std_or_gain: float = 0.02,
        ):
        super().__init__()
        self.encoder = DINOEncoder(model_size, patch_size=patch_size)
        self.decoder = Decoder(ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, resolution=256, z_channels=16)
        self.decoder.post_init(z_channels=z_channels)
        self.bottle_neck = MLP(in_dim=self.encoder.dim, out_dim=z_channels)

        init_weights(self.decoder.conv_in, conv_std_or_gain)
        init_weights(self.bottle_neck, conv_std_or_gain)
        init_weights(self.decoder, conv_std_or_gain)
            
    def forward(self, x, freeze_encoder=False, return_latent=False):
        ctx = torch.no_grad() if freeze_encoder else nullcontext()
        with ctx:
            latent_tokens = self.encoder(x)
        latent_tokens = self.bottle_neck(latent_tokens)
        x_rec = self.decoder(latent_tokens)
        if return_latent:
            return x_rec.float(), latent_tokens
        return x_rec.float()
    
    @torch.inference_mode()
    def encode(self, x):
        latent_tokens = self.encoder(x)
        latent_tokens = self.bottle_neck(latent_tokens)
        return latent_tokens
    
    @torch.inference_mode()
    def decode(self, latent_tokens):
        return self.decoder(latent_tokens)
    
    def load_pretrained(self, state_dict_path, ema=False):
        if not os.path.exists(state_dict_path):
            print(f'[WARNING] VAE state_dict_path {state_dict_path} not found, skip loading')
            return
        try:
            ckpt = torch.load(state_dict_path, map_location='cpu')
        except:
            ckpt = torch.load(state_dict_path, map_location='cpu', weights_only=False)
        if ema and "vae_ema" in ckpt:
            self.load_state_dict(ckpt['vae_ema'], strict=True)
        else:
            self.load_state_dict(ckpt['vae_wo_ddp'], strict=True)