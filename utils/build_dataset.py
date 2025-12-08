import os
from typing import Callable
from utils import dist
from utils.data import DataInfo
import PIL.Image as PImage
from torchvision.transforms import InterpolationMode, transforms
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from PIL import Image


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def center_crop_arr(pil_image, image_size=256):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_dataset(
    image_size: int = 256,
    mode='train', # ['train', 'val']
    mid_reso=1.5,
    epoch=0, iters=0,
    dataset_path: str = "",
    hflip: bool = True, # augmentation: horizontal, flip turn on this for imagenet
    workers: int = 16,
    same_seed_for_all_ranks: bool = False,
    global_bs: int = 256,
    local_bs: int = 32, 
    get_different_generator_for_each_rank: Callable = None,
):
    final_reso = image_size
    data_path = os.path.join(dataset_path, mode)
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    dataset = DatasetFolder(data_path, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug if mode == 'train' else val_aug)
        
    if mode == "train":
        sampler = DistInfiniteBatchSampler(
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            dataset_len=len(dataset),
            same_seed_for_all_ranks=same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            start_ep=epoch,
            start_it=iters,
            glb_batch_size=global_bs,
        )
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=workers,
            pin_memory=True,
            generator=get_different_generator_for_each_rank(),
            batch_sampler=sampler,
        )
        num_samples = len(dataset)
        num_batches = len(dataloader)
        return DataInfo(dataloader=iter(dataloader), num_samples=num_samples, num_batches=num_batches, sampler=sampler)
    else:
        sampler = EvalDistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        dataloader = DataLoader(
            dataset, num_workers=0, pin_memory=True, shuffle=False,
            batch_size=int(local_bs * 1.5), sampler=sampler,
        )
        return DataInfo(dataloader=dataloader, num_samples=len(dataset), num_batches=len(dataloader), sampler=sampler)


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class EvalDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        seps = np.linspace(0, len(dataset), num_replicas+1, dtype=int)
        beg, end = seps[:-1], seps[1:]
        beg, end = beg[rank], end[rank]
        self.indices = tuple(range(beg, end))
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed_for_all_rank=0, fill_last=False, shuffle=True, drop_last=False, start_ep=0, start_it=0):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = start_ep
        self.same_seed_for_all_ranks = seed_for_all_rank
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = self.start_ep
        while True:
            self.epoch += 1
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, same_seed_for_all_ranks=0, repeated_aug=0, fill_last=False, shuffle=True, start_ep=0, start_it=0):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        # print(f'global_max_p = iters_per_ep({self.iters_per_ep}) * glb_batch_size({self.glb_batch_size}) = {global_max_p}')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.fill_last:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        # global_indices = tuple(global_indices.numpy().tolist())
        
        seps = torch.linspace(0, global_indices.shape[0], self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank].item():seps[self.rank + 1].item()].tolist()
        self.max_p = len(local_indices)
        return local_indices