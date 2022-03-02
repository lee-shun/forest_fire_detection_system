import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class TBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TBlock, self).__init__()
        self.tblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.tblock(x)

class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_conv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.upconv(x)

if __name__ == '__main__':
    layer1 = Block(in_channels=3, out_channels=64)
    layer2 = TBlock(in_channels=128, out_channels=64)
    layer3 = Up_conv(in_channels=256, out_channels=128)
    feature_in = torch.randn((4, 3, 255, 255))
    feature_out = layer1(feature_in)
    print(feature_out.shape)