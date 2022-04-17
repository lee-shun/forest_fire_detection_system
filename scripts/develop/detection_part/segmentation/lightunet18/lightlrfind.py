# this function is to find appropriate learning rate for NNs
from torch_lr_finder import LRFinder
from lightunet import LightUnet
from lightCVloader import CVdataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

Image_hight = 400
Image_weight = 400
Atransform = A.Compose([
    A.Resize(Image_hight, Image_weight),
    A.HorizontalFlip(p = 0.2),
    A.RandomBrightnessContrast(p = 0.2),
    A.Normalize(
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
# the device used for training
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("=====>CUDA is available! Training on GPU...")
else:
    print("=====>CUDA is not available. Training on CPU...")

Img_dir = ('datasets/S_google_wildfire')
Mask_dir = ('datasets/S_google_wildfire_label')
Data = CVdataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = Atransform)

Model = LightUnet(in_channels=3, out_channels=1)
# Loss_fn = nn.MSELoss()
Loss_fn = nn.CrossEntropyLoss() # mask type `long`
Optimizer = optim.Adam(Model.parameters(), lr = 1e-4, weight_decay=1e-4)
# Optimizer = optim.SGD(Model.parameters(), lr=0.01, momentum=0.9)
batch_size = 2
counter = 0
data_loader = DataLoader(Data, batch_size = batch_size, 
                          num_workers = 0, 
                          pin_memory = True,
                          shuffle = True)
# remember to change the iteration to cover all data, 
# if the iteration is too short, there would be data not covered.
lr_finder = LRFinder(model = Model, optimizer=Optimizer, criterion=Loss_fn, device = Device)

lr_finder.range_test(train_loader=data_loader, end_lr = 1e-1, num_iter=10)
lr_finder.plot()
lr_finder.reset()

