import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class Attention_block(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(Attention_block, self).__init__()
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(in_channels),
        )
        self.phi = nn.Sequential(
            nn.ConvTranspose2d(gating_channels, in_channels, kernel_size= 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        xl = self.theta(x)
        _, _, H, W = xl.size()
        
        gl = F.upsample(self.phi(g), size = (H, W), mode = 'bilinear')
        # gl = self.phi(g)
        gl_size = gl.size()

        # print('xl_size:', (H, W), 'gl_size:', gl_size[2:])

        psi_in = self.relu(xl + gl)
        psi = self.psi(psi_in)
        y = torch.mul(x, psi)
        return y

if __name__ == '__main__':

    # batchsize = 2, channels = 128, inputsize = 255*255
    feature_x = torch.randn((2, 64, 255, 255))
    feature_g = torch.randn((2, 128, 128, 128))
    # model = Resnet34(img_channels=3, num_classes=3)
    attgate = Attention_block(in_channels=64, gating_channels=128)
    print(attgate.eval())
    feature_out = attgate(feature_x, feature_g)
    print('input shape:', feature_g.shape)
    print('up-sampling part shape:', feature_out.shape)

