import torch
import torch.nn as nn
from unet import RawUNet


class DarktableBayerWrapper(nn.Module):

    def __init__(self, base_model, default_alpha=0.002, default_beta=0.00005):
        super().__init__()
        self.base_model = base_model
        self.alpha = default_alpha
        self.beta = default_beta

    def forward(self, x):
        # Incoming from Darktable: (B, 4, H, W)
        # Synthesize internal 5th channel
        variance = self.alpha * x + self.beta
        sigma = torch.sqrt(torch.clamp(variance, min=1e-8))
        noise_map = sigma.mean(dim=1, keepdim=True)

        # Concatenate to 5 channels for U-Net
        x5 = torch.cat([x, noise_map], dim=1)
        return self.base_model(x5)


# 1. Load weights
device = torch.device("cpu")
raw_unet = RawUNet(in_channels=5, out_channels=4, base_channels=32).to(device)
ckpt = torch.load("best_raw_denoiser.pt", map_location=device)
state_dict = (
    ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
)
raw_unet.load_state_dict(state_dict)
raw_unet.eval()

wrapped_model = DarktableBayerWrapper(raw_unet).eval()

# 2. Export with all tensors embedded into model_bayer.onnx
dummy_input = torch.randn(1, 4, 512, 512, dtype=torch.float32)

torch.onnx.export(
    wrapped_model,
    dummy_input,
    "model_bayer.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
)

print("Export complete: model_bayer.onnx is 100% self-contained.")