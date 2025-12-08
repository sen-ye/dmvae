from PIL import Image
from PIL import ImageFile
from dataclasses import dataclass
from multiprocessing import Value
from typing import Iterator, Union
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.sampler import DistInfiniteBatchSampler


Image.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: Union[DataLoader, Iterator]
    num_samples: int = 0
    num_batches: int = 0
    sampler: Union[DistributedSampler, DistInfiniteBatchSampler] = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


def pil_loader(path):
    with open(path, 'rb') as f:
        img: Image.Image = Image.open(f).convert('RGB')
    return img


def pil_load(path: str):
    with open(path, 'rb') as f:
        img: Image.Image = Image.open(f).convert('RGB')
    return img