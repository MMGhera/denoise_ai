import rawpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import RawUNet


def denoise_raw_patch(
    raw_path: str,
    checkpoint_path: str = "best_raw_denoiser.pt",
    crop_size: int = 512,  # 512x512 Bayer crop (1024x1024 full res)
    iso_alpha: float = 0.005,  # Shot noise scale (higher for higher ISO)
    iso_beta: float = 0.0001,  # Read noise variance
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Load Model Architecture & Weights
    model = RawUNet(in_channels=5, out_channels=4, base_channels=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Ingest RAW File
    with rawpy.imread(raw_path) as raw:
        raw_bayer = raw.raw_image_visible.astype(np.float32)
        black_level = float(np.mean(raw.black_level_per_channel))
        white_level = float(raw.white_level)

        # Baseline camera demosaicing for visual reference
        noisy_rgb_full = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)

    # 3. Normalize to [0.0, 1.0] Linear Reflectance
    norm_raw = np.clip((raw_bayer - black_level) / (white_level - black_level), 0.0, 1.0)
    h, w = norm_raw.shape
    norm_raw = norm_raw[: h - (h % 2), : w - (w % 2)]

    # 4. Pack 2D Bayer Mosaic into 4-Channel Sub-Grids (RGGB)
    r = norm_raw[0::2, 0::2]
    g1 = norm_raw[0::2, 1::2]
    g2 = norm_raw[1::2, 0::2]
    b = norm_raw[1::2, 1::2]
    packed = np.stack([r, g1, g2, b], axis=0)  # Shape: (4, H/2, W/2)

    # 5. Extract Center Crop for Instant Inspection
    _, c_h, c_w = packed.shape
    start_y = (c_h - crop_size) // 2
    start_x = (c_w - crop_size) // 2
    packed_crop = packed[:, start_y : start_y + crop_size, start_x : start_x + crop_size]

    # 6. Synthesize 5th Noise Guidance Map
    variance = iso_alpha * packed_crop + iso_beta
    sigma = np.sqrt(np.clip(variance, 1e-8, None))
    noise_map = np.mean(sigma, axis=0, keepdims=True)

    input_5ch = np.concatenate([packed_crop, noise_map], axis=0)
    input_tensor = torch.from_numpy(input_5ch).unsqueeze(0).float().to(device)

    # 7. Model Forward Pass
    with torch.no_grad():
        denoised_4ch = model(input_tensor).squeeze(0).clamp(0.0, 1.0).cpu().numpy()

    # 8. Unpack Denoised Patch Back to 2D Mosaic
    denoised_mosaic = np.zeros((crop_size * 2, crop_size * 2), dtype=np.float32)
    denoised_mosaic[0::2, 0::2] = denoised_4ch[0]
    denoised_mosaic[0::2, 1::2] = denoised_4ch[1]
    denoised_mosaic[1::2, 0::2] = denoised_4ch[2]
    denoised_mosaic[1::2, 1::2] = denoised_4ch[3]

    noisy_mosaic = np.zeros((crop_size * 2, crop_size * 2), dtype=np.float32)
    noisy_mosaic[0::2, 0::2] = packed_crop[0]
    noisy_mosaic[0::2, 1::2] = packed_crop[1]
    noisy_mosaic[1::2, 0::2] = packed_crop[2]
    noisy_mosaic[1::2, 1::2] = packed_crop[3]

    # 9. Simple Linear Gamma RGB Approximation for Quick Preview
    def mosaic_to_preview_rgb(mosaic):
        # Bilinear channel pooling
        r_chan = mosaic[0::2, 0::2]
        g_chan = (mosaic[0::2, 1::2] + mosaic[1::2, 0::2]) * 0.5
        b_chan = mosaic[1::2, 1::2]
        rgb = np.stack([r_chan, g_chan, b_chan], axis=-1)
        # Apply standard display gamma curves for viewing
        return np.clip(np.power(rgb * 2.2, 1 / 2.2), 0.0, 1.0)

    noisy_rgb_crop = mosaic_to_preview_rgb(noisy_mosaic)
    denoised_rgb_crop = mosaic_to_preview_rgb(denoised_mosaic)

    # 10. Display Side-by-Side Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(noisy_rgb_crop)
    axes[0].set_title("Original Noisy RAW Crop", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(denoised_rgb_crop)
    axes[1].set_title("Denoised RAW Output (U-Net)", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Point this to any .NEF, .CR2, .DNG, or .ARW file
    denoise_raw_patch("./img_misc/_MMG5140.NEF")