import torch
import torch.nn as nn

class Squeezeblock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels_1, out_channels):
        super(Squeezeblock, self).__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)

        # expand
        self.expand_1 = nn.Conv2d(mid_channels, out_channels_1, 1, 1, 0)
        self.expand_3 = nn.Conv2d(mid_channels, out_channels - out_channels_1, 3, 1, 1)
    
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x_squezee = self.squeeze(x)
        x_squezee = self.relu(x_squezee)
        x_1 = self.expand_1(x_squezee)
        x_3 = self.expand_3(x_squezee)
        cat = torch.cat([x_1, x_3], dim = 1)
        x_out = self.relu(cat)
        return x_out

class UnSqueezeblock(nn.Module):
    def __init__(self, in_channels, mid_channels_1, mid_channels, out_channels):
        super(UnSqueezeblock, self).__init__()

        self.expand_1 = nn.Conv2d(in_channels, mid_channels_1, 1, 1, 0)
        self.expand_3 = nn.Conv2d(in_channels, mid_channels - mid_channels_1, 3, 1, 1)
        self.squeeze = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)
        self.outconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.expand_1(x)
        x_3 = self.expand_3(x)
        x_mid = torch.cat([x_1, x_3], dim = 1)
        x_out = self.squeeze(x_mid)
        x_out = self.relu(x_out)
        x_out = self.outconv(x_out)
        x_out = self.relu(x_out)
        return x_out
    
if __name__ == "__main__":
        # batchsize = 1, channels = 64, inputsize = 255*255
    feature = torch.randn((1, 64, 255, 255))

    model = Squeezeblock(in_channels = 64, mid_channels = 75, out_channels_1 = 15, out_channels = 128)
    print(model.eval())
    preds = model(feature)
    print('input shape:', feature.shape)
    print('preds shape:', preds.shape)

    model = UnSqueezeblock(in_channels = 64, mid_channels = 55, mid_channels_1 = 15, out_channels = 32)
    print(model.eval())
    uppreds = model(feature)
    print('input shape:', feature.shape)
    print('uppreds shape:', uppreds.shape)