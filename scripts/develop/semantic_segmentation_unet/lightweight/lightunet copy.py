import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torchvision.transforms.functional as TF
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/qiao/dev/giao/havingfun/deving/common')
import torchvision.transforms as T
from inout import Inputlayer, Outlayer
from attentiongate import Attention_block
from doubleconv import Block, TBlock, Up_conv
from depthwise import DDepthwise, UDepthwise

class LightUnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, scale_factor = 1):
        super(LightUnet, self).__init__()
        num = np.array([2, 2, 2, 2])
        filters = np.array([64, 128, 256, 512])
        filters = filters // scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # down-sampling
        self.Conv0 = Inputlayer(in_channels, filters[0])

        self.down1 = nn.Sequential(
                DDepthwise(filters[0], filters[0]),
                UDepthwise(filters[0], filters[0])
                )
        self.down2 = nn.Sequential(
                DDepthwise(filters[0], filters[1]),
                UDepthwise(filters[1], filters[1])
                )
        self.down3 = nn.Sequential(
                DDepthwise(filters[1], filters[2]),
                UDepthwise(filters[2], filters[2])
                )
        self.neck = nn.Sequential(
                DDepthwise(filters[2], filters[3]),
                UDepthwise(filters[3], filters[3])
                )

        # up_sampling
        self.Up3 = Up_conv(filters[3], filters[2])
        self.Att3 = Attention_block(filters[2], filters[3])
        self.up_conv3 = TBlock(filters[3], filters[2])

        self.Up2 = Up_conv(filters[2], filters[1])
        self.Att2 = Attention_block(filters[1], filters[2])
        self.up_conv2 = TBlock(filters[2], filters[1])

        self.Up1 = Up_conv(filters[1], filters[0])
        self.Att1 = Attention_block(filters[0], filters[1])
        self.up_conv1 = TBlock(filters[1], filters[0])

        # self.up_conv0 = Up_conv(filters[0], out_channels)
        self.outlayer = Outlayer(filters[0], out_channels)

    def forward(self, x):
        x0 = self.Conv0(x)
        # print('input-c64 size :', x0.size())
        x1 = self.down1(x0)
        # print('c64-c64 size:', x1.size())
        x2 = self.Maxpool(x1)
        x2 = self.down2(x2)
        # print('c64-c128 size:', x2.size())
        x3 = self.Maxpool(x2)
        x3 = self.down3(x3)
        x_neck = self.Maxpool(x3)
        # print('c128-c256 size:', x3.size())
        x_neck = self.neck(x_neck)
        # print('c256-c512 neck size:', x_neck.size())
        gate3 = self.Att3(x3, x_neck)
        # print(gate3.size())     
        _up3 = self.Up3(x_neck)
        _up3 = TF.resize(_up3, size = gate3.shape[2:])
        # print(_up3.size())
        up3 = torch.cat((gate3, _up3), 1)
        up3 = self.up_conv3(up3)

        gate2 = self.Att2(x2, up3)
        _up2 = self.Up2(up3)
        _up2 = TF.resize(_up2, size = gate2.shape[2:])
        up2 = torch.cat((gate2, _up2), 1)
        up2 = self.up_conv2(up2)

        gate1 = self.Att1(x1, up2)
        _up1 = self.Up1(up2)
        _up1 = TF.resize(_up1, size = gate1.shape[2:])
        up1 = torch.cat((gate1, _up1), 1)
        up1 = self.up_conv1(up1)
        # up0 = self.up_conv0(up1)
        out = self.outlayer(up1)

        return out

if __name__ == '__main__':
    # batchsize = 4, channels = 3, inputsize = 400*400
    img = torch.randn((4, 3, 400, 400))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = LightUnet(in_channels=3, out_channels = 1)
    print(model.eval())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The depthwise seperable convolution uses {params} parameters.')
    preds = model(img)
#     process = T.Resize(img.size()[2])
#     preds = process(preds)
    print('input shape:', img.size())
    print('preds shape:', preds.size())






