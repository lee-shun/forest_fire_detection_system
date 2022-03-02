# 1 kernel works for one channel
# deep wise 3x3: n_kernels = in_channels, then for every filter nn.Conv2d(1, out_channels, 3, 3, 1)
# num_parameter for each filter: 3x3x1xout_channels
# point wise 1x1: num_parameter for each filter: 1x1xin_channelsxout_channels

import torch
import torch.nn as nn

class DDepthwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDepthwise, self).__init__()
        self.ddepthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups = in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups = in_channels),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        y = self.ddepthwise(x)
        return y

class UDepthwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UDepthwise, self).__init__()
        self.udepthwise = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, groups = in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, groups = in_channels),
            nn.ConvTranspose2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.udepthwise(x)
        return y

if __name__ == '__main__':
    block = DDepthwise(in_channels=64, out_channels=128)
    print(block.eval())
    feature_in = torch.randn((4, 64, 255, 255))
    params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f'The depthwise seperable convolution uses {params} parameters.')
    feature_out = block(feature_in)
    print(feature_in.size())
