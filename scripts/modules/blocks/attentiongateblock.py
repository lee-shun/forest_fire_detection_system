# 2022-04-03
# Re-modifying
# gate signal: in upsampling 
# attention signal: in downsampling, one more down.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from depthwiseblock import DDepthwise, UDepthwise, Up_conv

import sys
sys.path.insert(1, 'havingfun/deving/tools')
from resizetensor import sizechange

class Attentiongate_block(nn.Module):
    def __init__(self, att_channels, gating_channels):
        super(Attentiongate_block, self).__init__()
        self.Att = nn.Sequential(
            nn.Conv2d(att_channels, gating_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(gating_channels),
            # to change H,W of feature map
            nn.Conv2d(gating_channels, gating_channels, 1, 2, 0), 
            nn.BatchNorm2d(gating_channels),
        )

        self.Gate = nn.Sequential(
            nn.ConvTranspose2d(gating_channels, gating_channels, kernel_size= 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(gating_channels),
        )
        self.ReLU = nn.ReLU(inplace=True)

        # to change H, W of feature map
        self.upsample = nn.Upsample(scale_factor=2)
        self.psi = nn.Sequential(
            nn.ConvTranspose2d(gating_channels, att_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(att_channels),
            nn.Sigmoid(),
        )

    def forward(self, att_in, gate):
        att_out = self.Att(att_in)
        # print(f'att_out size: {att_out.size()}')

        gate = self.Gate(gate)
        # print(f'gating size: {gate.size()}')
        # if gate.size != att.size():
        gate = sizechange(gate, att_out)
        # print(f'resized gating size: {gl.size()}')

        psi_in = self.ReLU(att_out + gate)
        # print(f'psi_in size: {psi_in.size()}')
        psi_out = self.psi(psi_in)
        if psi_out.size() != gate.size():
            psi_out = sizechange(psi_out, gate)
        # print(f'psi_out size: {psi_out.size()}')
        y = self.upsample(psi_out)
        y = sizechange(y, att_in)
        y = torch.mul(y, att_in)
        # print(f'attention gate out size: {y.size()}')
        return y

if __name__ == '__main__':

    tensor_in = torch.randn((4, 64, 99, 99))
    down01 = tensor_in
    pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
    down02 = pool_layer(tensor_in)
    att0 = down01
    print(f'down02, att0 size: {down02.size()}')

    Down10 = DDepthwise(64, 128)
    down10 = Down10(down02)
    Down11 = DDepthwise(128, 128)
    down11 = Down11(down10)
    gate1 = down11

    Att1 = Attentiongate_block(64, 128)
    _up10 = Att1(att0, gate1)
    up_conv1 = Up_conv(128, 64)
    _up11 = up_conv1(down11)
    _up11 = sizechange(_up11, _up10)
    _up1 = torch.cat((_up10, _up11), 1)
    Up1 = UDepthwise(128, 64)
    up1 = Up1(_up1)
    print(f'up1 size: {up1.size()}')