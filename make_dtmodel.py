import zipfile
import os

def create_dtmodel(onnx_path: str, manifest_path: str, output_path: str):
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, arcname="model.onnx")
        zf.write(manifest_path, arcname="config.json")
    print(f"Successfully created package: {output_path}")

if __name__ == "__main__":
    create_dtmodel("custom_raw_unet.onnx", "config.json", "custom_raw_unet.dtmodel")