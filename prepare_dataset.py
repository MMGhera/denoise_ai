import os
import glob
import argparse
from pathlib import Path
import numpy as np
import rawpy
import h5py
from tqdm import tqdm

def get_canonical_rggb(raw_cfa: np.ndarray, pattern: np.ndarray, color_desc: str) -> np.ndarray:
    bayer_layout = "".join([color_desc[idx] for row in pattern for idx in row])

    s00 = raw_cfa[0::2, 0::2]
    s01 = raw_cfa[0::2, 1::2]
    s10 = raw_cfa[1::2, 0::2]
    s11 = raw_cfa[1::2, 1::2]

    if bayer_layout == "RGGB":
        r, g1, g2, b = s00, s01, s10, s11
    elif bayer_layout == "BGGR":
        b, g1, g2, r = s00, s01, s10, s11
    elif bayer_layout == "GRBG":
        g1, r, b, g2 = s00, s01, s10, s11
    elif bayer_layout == "GBRG":
        g1, b, r ,g2 = s00, s01, s10, s11
    else:
        r, g1, g2, b = s00, s01, s10, s11

    return np.stack([r, g1, g2, b], axis=0)

def process_raw_file(file_path: str, patch_size: int, stride: int) -> list:
    try:
        with rawpy.imread(file_path) as raw:
            raw_image = raw.raw_image_visible.astype(np.float32)
            black_level = float(np.mean(raw.black_level_per_channel))
            white_level = float(raw.white_level)
            pattern = raw.raw_pattern
            color_desc = raw.color_desc.decode("latin1")

    except Exception as e:
        print(f"\n[Warning] Could not read {file_path}: {e}")
        return []

    # 1. Normalize sensor values into [0.0, 1.0] linear reflectance
    norm_image = np.clip((raw_image - black_level) / (white_level - black_level), 0.0, 1.0)

    # 2. Crop off odd trailing pixel row/column so 2x2 Bayer slicing has matching shapes
    h, w = norm_image.shape
    norm_image = norm_image[: h - (h % 2), : w - (w % 2)]

    # 3. Pack 2D mosaic to (4, H/2, W/2)
    packed = get_canonical_rggb(norm_image, pattern, color_desc)

    # Slicing parameters for the packed 4-channel tensor
    c, h_packed, w_packed = packed.shape
    p_half = patch_size // 2
    s_half = stride // 2

    patches = []

    for y in range(0, h_packed - p_half + 1, s_half):
        for x in range(0, w_packed - p_half + 1, s_half):
            crop = packed[:, y : y + p_half, x : x + p_half]

            # Quality Filter: Discard completely clipped highlights or dead shadows
            mean_val = np.mean(crop)
            if mean_val < 0.002 or mean_val > 0.98:
                continue

            patches.append(crop.astype(np.float32))

    return patches

def build_dataset(input_dir: str, output_h5: str, patch_size: int = 256, stride: int = 256):
    input_path = Path(input_dir)
    extensions = ("*.NEF", "*.nef", "*.CR2", "*.cr2", "*.CR3", "*.cr3", "*.ARW", "*.arw", "*.DNG", "*.dng")

    raw_files = []
    for ext in extensions:
        raw_files.extend(glob.glob(str(input_path / "**" / ext), recursive=True))

    raw_files = sorted(list(set(raw_files)))
    print(f"Found {len(raw_files)} files")
    if not raw_files:
        return

    p_half = patch_size // 2

    with h5py.File(output_h5, "w") as h5_file:
        dset = h5_file.create_dataset(
            "bayer_patches",
            shape = (0, 4, p_half, p_half),
            maxshape=(None, 4, p_half, p_half),
            dtype = "float32",
            chunks = (64, 4, p_half, p_half),
            compression = "gzip",
            compression_opts = 4,
        )

        buffer = []
        buffer_size = 500

        for file_path in tqdm(raw_files, desc="Processing RAW files"):
            patches = process_raw_file(file_path, patch_size, stride)
            buffer.extend(patches)

            if len(buffer) >= buffer_size:
                current_len = dset.shape[0]
                dset.resize((current_len + len(buffer), 4, p_half, p_half))
                dset[current_len :] = np.stack(buffer, axis=0)
                buffer.clear()

        if buffer:
            current_len = dset.shape[0]
            dset.resize((current_len + len(buffer), 4, p_half, p_half))
            dset[current_len :] = np.stack(buffer, axis=0)
            buffer.clear()

        print(f"\nCompleted. Saved {dset.shape[0]} patches to {output_h5}")
        print(f"Dataset Shape: {dset.shape} (N, Channels, H, W)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 4-channel Bayer patches from RAW files to HDF5.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing RAW files.")
    parser.add_argument("--output", type=str, default="dataset.h5", help="Path to output HDF5 file.")
    parser.add_argument("--patch_size", type=int, default=256, help="2D sensor patch size (default: 256 -> 128x128 packed).")
    parser.add_argument("--stride", type=int, default=256, help="Stride between patches (use lower for overlapping crops).")

    args = parser.parse_args()
    build_dataset(args.input_dir, args.output, args.patch_size, args.stride)