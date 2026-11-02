import torch
from unet import RawUNet

device = torch.device("cpu")

# 1. Initialize architecture and load checkpoint
model = RawUNet(in_channels=5, out_channels=4, base_channels=32).to(device)
checkpoint = torch.load("best_raw_denoiser.pt", map_location=device)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# 2. Dummy input with dynamic spatial dimensions (B=1, C=5, H=256, W=256)
dummy_input = torch.randn(1, 5, 256, 256, dtype=torch.float32)

# 3. Export to ONNX with dynamic spatial axes
output_onnx_path = "custom_raw_unet.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_onnx_path,
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

print(f"Model exported successfully to {output_onnx_path}")