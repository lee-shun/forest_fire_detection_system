# Every 3x3 conv kernel works for one channel of input tensor
# deep wise 3x3: n_kernels = in_channels, then for every filter nn.Conv2d(1, out_channels, 3, 3, 1)
# num_parameter for each filter: 3x3x1xout_channels
# point wise 1x1: num_parameter for each filter: 1x1x in_channels x out_channels

import torch
import torch.nn as nn

class DDepthwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDepthwise, self).__init__()
        self.ddepthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups = int(in_channels)),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups = int(out_channels)),
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
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, groups = int(in_channels)),
            nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, groups = int(out_channels)),
            nn.ConvTranspose2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.udepthwise(x)
        return y

class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_conv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.upconv(x)

if __name__ == '__main__':
    block = Up_conv(in_channels=128, out_channels=64)
    print(block.eval())
    feature_in = torch.randn((4, 128, 400, 400))
    params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f'The depthwise seperable convolution uses {params} parameters.')
    feature_out = block(feature_in)
    print(feature_in.size())
    print(feature_out.size())
