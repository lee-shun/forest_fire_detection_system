import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


# input conv
class Inputlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inputlayer, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        return self.conv0(x)

class Outlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outlayer, self).__init__()
        # 1x1 Conv
        self.convf = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()
        self.outscale = nn.Upsample(scale_factor=2)
        self.convo = nn.ConvTranspose2d(in_channels, out_channels, 7, 2, 3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.convf(x)
        y = self.relu(y)
        y = self.outscale(x)
        y = self.convo(y)
        y = self.sigmoid(y)
        return y

if __name__ == "__main__":
    layer1 = Inputlayer(in_channels=3, out_channels=64)
    layer2 = Outlayer(in_channels=64, out_channels=1)
    feature_in = torch.randn((4, 64, 99, 99))
    feature_out = layer2(feature_in)
    print(feature_out.shape)