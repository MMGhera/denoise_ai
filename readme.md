# Custom RAW U-Net Denoiser

A custom 5-channel U-Net trained in PyTorch to denoise RAW Bayer sensor data (RGGB). The model is conditioned on a synthetic Poisson-Gaussian noise map to recover extreme high-ISO captures and is wrapped to output residual noise ($\Delta$) for native integration into Darktable's AI pipeline.

## Features
* **Bayer-Level Processing:** Operates directly on the RAW mosaic before demosaicing to preserve micro-contrast and prevent artifact amplification.
* **Darktable Integration:** Packaged as a `.dtmodel` archive containing a fully self-contained ONNX graph and manifest for Darktable's `rawdenoise` module.
* **Residual Subtraction:** Outputs the noise delta rather than the absolute image, aligning with Darktable's subtractive engine and preventing color channel inversion.

## Prerequisites
To run local training, inference, or model conversion, install the required dependencies:
```bash
pip install torch rawpy onnx numpy matplotlib
```

## Darktable Installation
1. Download `custom_raw_unet.dtmodel` from the **Releases** tab.
2. Open Darktable and navigate to **Preferences $\to$ AI Models**.
3. Click **import from file...** and select the downloaded `.dtmodel` archive.
4. In the Darkroom view, activate the **raw denoise** module and select the custom U-Net model.

## Local PyTorch Inference
You can test the raw PyTorch weights (`best_raw_denoiser.pt`) directly on a RAW file (e.g., `.NEF`, `.DNG`) outside of Darktable. Adjust the `iso_alpha` and `iso_beta` parameters inside the script to scale the denoising strength for different ISO ranges.

```bash
python test_denoise.py
```

## Repository Structure
* **`unet.py`**: Core 5-channel U-Net model architecture definition.
* **`test_denoise.py`**: Local inference script with side-by-side Matplotlib visualizer for instant patch inspection.
* **`export_self_contained.py`**: Wraps the 5-channel PyTorch model to accept a 4-channel ONNX input, embeds all layer weights natively to avoid external data paths, and converts the output to a residual delta.
* **`make_dtmodel.py`**: Packages the compiled `.onnx` graph and manifest into the final `.dtmodel` archive.
* **`config.json`**: Darktable manifest dictating the `bayer_v1` layout, 512 input size, and force_rggb CFA pattern.
* **`prepare_dataset.py`** & **`noise_gen.py`**: Pipeline scripts for slicing datasets and synthesizing Poisson-Gaussian noise during training.