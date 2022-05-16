import os
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import glob
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

class CVdataset(Dataset):
    def __init__(self,  img_dir= 'img_dir', mask_dir = 'mask_dir', transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)
  
    def __len__(self):     
        return len(self.imgs)
        

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, 'Label_' + self.imgs[index])
        img_np = cv2.imread(img_path)
        # print(img_np.shape)
        # convert to original image channels, because cv2.imread may change it
        img_np = img_np[..., ::-1] 
        mask_np = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        # mask_np = mask_np[..., ::-1]
        # print(f'mask_np.shape: {mask_np.shape}')
        
        # there are multiple classes for segmentation, then no need 
        # mask_np[mask_np > 0.0] = 1.0

        if self.transform:           
            augmentations = self.transform(image = img_np, mask = mask_np)
            img_tensor = augmentations['image']
            mask_tensor = augmentations['mask'].float()
        return img_tensor, mask_tensor

if __name__ == '__main__':
    Img_dir = ('datasets/S_google_wildfire')
    Mask_dir = ('datasets/S_google_wildfire_label')
    data = CVdataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = Atransform)

    # split into train dataset and validation dataset
    dataset_size = len(data)
    print(f"Total number of images: {dataset_size}")
    valid_split = 0.2
    valid_size = int(valid_split*dataset_size)
    indices = torch.randperm(len(data)).tolist()
    train_data = Subset(data, indices[:-valid_size])
    val_data = Subset(data, indices[-valid_size:])
    print(f"Total training images: {len(train_data)}")
    print(f"Total valid_images: {len(val_data)}")

    batch_size = 1
    counter = 0
    data_loader = DataLoader(data, batch_size = batch_size, 
                          num_workers = 0, 
                          pin_memory = 2,
                          shuffle = True)

    for j, data in tqdm(enumerate(data_loader), total = len(data) // batch_size):
        counter += 1
        img_tensor, mask_tensor = data
    print('img_tensor size:', img_tensor.permute(0, 2, 3, 1).size())
    print('mask_tensor size:', mask_tensor.size())
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img_tensor.permute(0, 2, 3, 1).squeeze(0))
    ax[1].imshow(mask_tensor.squeeze(0))
    plt.show()


