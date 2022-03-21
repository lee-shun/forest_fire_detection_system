# 2022-03-14
# this block is used to change the tensor size
# espicially for the tensor passing through up-sampling part
# where the up-sampling out need to be concatenated to the gate skip connection output
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

def sizechange(input_tensor, gate_tensor):
    sizechange = nn.UpsamplingNearest2d(size = gate_tensor.shape[2:])
    out = sizechange(input_tensor)
    return out

if __name__ == "__main__":
    input_tensor = torch.randn((4, 64, 395, 395))
    gate_tensor = torch.randn((4, 1, 400, 400))
    out = sizechange(input_tensor, gate_tensor)
    print('input tensor size:', input_tensor.size())
    print('gate tensor size:', gate_tensor.size())
    print('out_tensor size:', out.size())