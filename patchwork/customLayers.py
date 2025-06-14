import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, nD=2):
    if nD == 3:
        conv = nn.Conv3d
        bn = nn.BatchNorm3d
    else:
        conv = nn.Conv2d
        bn = nn.BatchNorm2d
    return nn.Sequential(
        conv(in_ch, out_ch, kernel_size=3, padding=1),
        bn(out_ch),
        nn.ReLU(inplace=True)
    )


def up_block(in_ch, out_ch, nD=2):
    if nD == 3:
        convT = nn.ConvTranspose3d
    else:
        convT = nn.ConvTranspose2d
    return nn.Sequential(
        convT(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm3d(out_ch) if nD == 3 else nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class CNNblock(nn.Module):
    """A simple sequential block wrapper."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        if len(layers) > 0 and hasattr(layers[-1], 'out_channels'):
            self.out_channels = layers[-1].out_channels

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def create_unet(depth=4, out_channels=1, feature_dim=16, nD=2):
    layers = []
    in_c = 1
    fdim = feature_dim
    for _ in range(depth):
        layers.append(conv_block(in_c, fdim, nD=nD))
        in_c = fdim
        fdim *= 2
    if nD == 3:
        layers.append(nn.Conv3d(in_c, out_channels, kernel_size=1))
    else:
        layers.append(nn.Conv2d(in_c, out_channels, kernel_size=1))
    return CNNblock(layers)


def simple_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, num_classes)
    )


__all__ = [
    'conv_block',
    'up_block',
    'CNNblock',
    'create_unet',
    'simple_classifier',
]
