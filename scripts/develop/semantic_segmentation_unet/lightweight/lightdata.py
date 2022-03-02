# data loading part
import os
from PIL import Image
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
# import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

Image_hight =400
Image_weight = 400
transform = A.Compose([
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
# transform = T.Compose([
#     T.Resize((Image_hight, Image_weight)),
#     T.ToTensor(), 
#     T.Normalize(
#         mean = [0.0, 0.0, 0.0],
#         std = [1.0, 1.0, 1.0],
#     ),
#     ])

class JinglingDataset(Dataset):
    def __init__(self,  img_dir, mask_dir, transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):        
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        img_im = Image.open(img_path).convert('RGB')
        img_np = np.array(img_im)
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_np[mask_np > 0.0] = 1.0
        if self.transform:           
            augmentations = self.transform(image = img_np, mask = mask_np)
            img_tensor = augmentations['image']
            mask_tensor = augmentations['mask'].long()
        return img_tensor, mask_tensor

if __name__ == '__main__':
    Img_dir = ('dataset/imgs/jinglingseg/images')
    Mask_dir = ('dataset/imgs/jinglingseg/masks')
    data = JinglingDataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = transform)
    # img, mask = data
    # print(img.shape)
    for i in range(len(data)):
        img, mask = data[i][0], data[i][1]
    print(mask[1].shape)

    dataset_size = len(data)
    print(f"Total number of images: {dataset_size}")
    valid_split = 0.2
    valid_size = int(valid_split*dataset_size)
    indices = torch.randperm(len(data)).tolist()
    train_data = Subset(data, indices[:-valid_size])
    val_data = Subset(data, indices[-valid_size:])
    print(f"Total training images: {len(train_data)}")
    print(f"Total valid_images: {len(val_data)}")
    plt.imshow(train_data[3][1])
    plt.show()




