import sys
import rawpy

def inspect_raw(file_path: str) -> None:
    with rawpy.imread(file_path) as raw:
        color_desc = raw.color_desc.decode('latin1')
        pattern_matrix = raw.raw_pattern

        bayer_2x2 = [[color_desc[idx] for idx in row] for row in pattern_matrix]
        bayer_string = "".join("".join(row) for row in bayer_2x2)
        print(f"File: {file_path}")
        print("_"*40)
        print(f"Raw sensor dimensions:          {raw.raw_image.shape} (H x W)")
        print(f"Active/visible area:            {raw.raw_image_visible.shape} (W x H)")
        print(f"Bayer CFA pattern:              {bayer_string} ({bayer_2x2})")
        print(f"Black level per channel:        {raw.black_level_per_channel}")
        print(f"White (saturation) level:       {raw.white_level}")
        print(f"Camera white balance (as shot): {raw.camera_whitebalance}")
        print(f"Daylight WB multipliers:        {raw.daylight_whitebalance}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_raw.py <path_to_file>")
        sys.exit(1)

    inspect_raw(sys.argv[1])