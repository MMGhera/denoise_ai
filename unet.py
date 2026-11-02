import torch
import torch.nn.functional as F
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(out_channels * 2, out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.reduce_conv(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class RawUNet(nn.Module):
    def __init__(self, in_channels: int =5, out_channels: int =4, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noisy_bayer = x[:, :self.out_channels, :, :]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)

        residual = self.outc(d)

        return noisy_bayer + residual

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = RawUNet(in_channels=5, out_channels=4, base_channels=32).to(device)

    dummy_input = torch.randn(16, 5, 128, 128).to(device)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model successfully initialized on {device}")
    print(f"Input Shape:        {dummy_input.shape}")
    print(f"Output Shape:       {output.shape} (Expected: [16, 4, 128, 128])")
    print(f"Trainable Params:   {total_params:,} (~{total_params / 1e6:.2f}M)")