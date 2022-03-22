# this function is to find appropriate learning rate for NNs
from torch_lr_finder import LRFinder
from lightunet import LightUnet
from lightCVloader import CVdataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm

Image_hight =400
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
Img_dir = ('datasets/S_kaggle_wildfire')
Mask_dir = ('datasets/S_kaggle_wildfire_label')
Data = CVdataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = Atransform)

Model = LightUnet(in_channels=3, out_channels=1)
Loss_fn = nn.MSELoss()
Optimizer = optim.Adam(Model.parameters(), lr = 1e-4, weight_decay = 1e-2)
batch_size = 1
counter = 0
data_loader = DataLoader(Data, batch_size = batch_size, 
                          num_workers = 0, 
                          pin_memory = 2,
                          shuffle = True)

lr_finder = LRFinder(model = Model, optimizer=Optimizer, criterion=Loss_fn, device = Device)
lr_finder.range_test(train_loader=data_loader, end_lr = 50, num_iter=50)
lr_finder.plot()
plt.show()
lr_finder.reset()
