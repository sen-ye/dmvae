from typing import Tuple
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import torch.nn.functional as F
import os


def get_fid_is(dir_raw: str=None, dir_gen: str=None, 
        fid_statistics_file=None, inception_weights_path=None) -> Tuple[float, float]:
    import torch_fidelity # type: ignore
    # NOTE: This is an updated version of torch_fidelity, to install this version, please run
    # pip3 install git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity
    # For IMAGENET statistics, refer to https://github.com/LTH14/mar/tree/main/fid_stats
    # For inception weights, refer to https://github.com/mseitzer/pytorch-fid/releases
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dir_gen,
        input2=None,
        samples_shuffle=True,
        samples_find_deep=True,
        samples_find_ext='png,jpg,jpeg',
        samples_ext_lossy='jpg,jpeg',
        fid_statistics_file=fid_statistics_file,
        cuda=True,
        batch_size=1536,
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
        feature_layer_fid='2048',
        feature_layer_kid='2048',
        feature_extractor_weights_path=inception_weights_path,
        verbose=True,

        save_cpu_ram=False,
        rng_seed=0,
    )
    fid = metrics_dict['frechet_inception_distance']
    isc = metrics_dict['inception_score_mean']
    return fid, isc


class FID(FrechetInceptionDistance):
    def __init__(self, feature=2048, eval_size=299, normalize=False):
        super().__init__(feature=feature, normalize=normalize)
        self.eval_size = eval_size
        self.normalize = normalize

    def update(self, imgs, real,):
        '''
        Input: imgs should have value range [0, 1]
        '''
        if imgs.size(-1) != self.eval_size:
            imgs = F.interpolate(imgs, size=(self.eval_size, self.eval_size), mode="bicubic",)
        if not self.normalize:
            imgs = (imgs * 255).clamp(0, 255).to(dtype=torch.uint8)
        return super().update(imgs, real)
    
    def save(self, real, path):
        if real:
            to_save_dict = {
                "real_features_sum": self.real_features_sum,
                "real_features_cov_sum": self.real_features_cov_sum,
                "real_features_num_samples": self.real_features_num_samples,
            }
        else:
            to_save_dict = {
                "fake_features_sum": self.fake_features_sum,
                "fake_features_cov_sum": self.fake_features_cov_sum,
                "fake_features_num_samples": self.fake_features_num_samples,
            }
        print("Saving FID stats to", path)
        for k, v in to_save_dict.items():
            print(k, v.shape)
        torch.save(to_save_dict, path)

    def load(self, real, path):
        loaded_dict = torch.load(path, map_location="cpu")
        device = self.real_features_sum.device
        if real:
            self.real_features_sum = loaded_dict["real_features_sum"].to(device)
            self.real_features_cov_sum = loaded_dict["real_features_cov_sum"].to(device)
            self.real_features_num_samples = loaded_dict["real_features_num_samples"].to(device)
        else:
            self.fake_features_sum = loaded_dict["fake_features_sum"].to(device)
            self.fake_features_cov_sum = loaded_dict["fake_features_cov_sum"].to(device)
            self.fake_features_num_samples = loaded_dict["fake_features_num_samples"].to(device)