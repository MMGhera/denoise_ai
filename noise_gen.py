import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RawDenoiseDataset(Dataset):
    def __init__(self, h5_path: str, is_train: bool = True):
        self.h5_path = h5_path
        self.is_train = is_train

        with h5py.File(self.h5_path, "r") as f:
            self.length = f["bayer_patches"].shape[0]

        self.h5_file = None
        self.patches = None

    def _init_h5(self):
        self.h5_file = h5py.File(self.h5_path, "r")
        self.patches = self.h5_file["bayer_patches"]

    def _sample_noise_params(self):
        log_alpha = np.random.uniform(-4.0, -1.5)
        alpha = 10.0 ** log_alpha

        log_beta = 1.8 * log_alpha + 2.0 +np.random.uniform(-0.3, 0.3)
        beta = 10.0 ** log_beta

        return float(alpha), float(beta)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[-2])
        if torch.rand(1).item() > 0.5:
            patch = torch.rot90(patch, k=1, dims=[-2, -1])

        return patch

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if self.patches is None:
            self._init_h5()

        clean_np = self.patches[idx]
        clean_tensor = torch.from_numpy(clean_np).float()

        if self.is_train:
            clean_tensor = self._augment(clean_tensor)

        alpha, beta = self._sample_noise_params()

        variance = alpha * clean_tensor + beta
        sigma = torch.sqrt(torch.clamp(variance, min=1e-8))

        noise = torch.randn_like(clean_tensor)
        noisy_tensor = torch.clamp(clean_tensor + noise * sigma, 0.0, 1.0)

        noise_map = sigma.mean(dim=0, keepdim=True)

        input_tensor = torch.cat([noisy_tensor, noise_map], dim=0)

        return {
            "input": input_tensor,
            "target": clean_tensor,
            "alpha": alpha,
            "beta": beta
        }

#----------------------------------------------------------

# if __name__ == "__main__":
#     dataset = RawDenoiseDataset("./dataset.h5", is_train=True)
#     loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
#
#     batch = next(iter(loader))
#     inputs = batch["input"]
#     targets = batch["target"]
#
#     print(f"Batch Input Shape:   {inputs.shape}  -> Expected: [16, 5, 128, 128]")
#     print(f"Batch Target Shape:  {targets.shape} -> Expected: [16, 4, 128, 128]")
#     print(f"Sample Alpha:        {batch['alpha'][0]:.6f}")
#     print(f"Sample Beta:         {batch['beta'][0]:.8f}")