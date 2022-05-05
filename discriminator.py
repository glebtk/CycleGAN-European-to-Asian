import torch
import torch.nn as nn

from torchsummary import summary

import config


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]  # 64 по дефолту
        for feature in features[1:]:
            # На последнем шаге stride=1, на предыдущих stride=2
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)  # Первый слой
        return torch.sigmoid(self.model(x))  # Добавляем остальные слои


def test():
    x = torch.randn((config.BATCH_SIZE, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE))
    model = Discriminator(config.IN_CHANNELS)
    prediction = model(x)

    print("Input shape: ", x.shape)
    print("Output shape: ", prediction.shape)

    # summary(model, depth=5)


if __name__ == "__main__":
    test()
