import torch
import torch.nn as nn

from torchsummary import summary

import config


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlockDown(channels, channels, kernel_size=3, padding=1),
            ConvBlockDown(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlockDown(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlockDown(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlockUp(num_features*4, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlockUp(num_features*2, num_features*1, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, in_channels, kernel_size=6, stride=1, padding=4, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last(x)
        return torch.tanh(x)


def test():
    x = torch.randn((config.BATCH_SIZE, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE))
    model = Generator(config.IN_CHANNELS, num_residuals=9)
    prediction = model(x)

    print("Input shape: ", x.shape)
    print("Output shape: ", prediction.shape)

    # summary(model, depth=5)


if __name__ == "__main__":
    test()




