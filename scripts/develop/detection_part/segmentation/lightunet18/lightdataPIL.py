# data loading part
import os
from PIL import Image
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm
import torchvision

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
        # mask_path = os.path.join(self.mask_dir, self.masks[index])
        mask_path = os.path.join(self.mask_dir, 'label_' + self.imgs[index])
        mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print(mask_np.tolist())
        # mask_np[mask_np > 0.0] = 1.0
        if self.transform:           
            augmentations = self.transform(image = img_np, mask = mask_np)
            img_tensor = augmentations['image']
            # print(img_tensor.size()
            mask_tensor = augmentations['mask'].float()
            # print(mask_tensor)
        return img_tensor, mask_tensor
    
if __name__ == '__main__':
    Img_dir = ('datasets/S_google_wildfire')
    Mask_dir = ('datasets/S_google_wildfire_label')
    Data = JinglingDataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = Atransform)
    # img, mask = data
    # print(img.shape)
    # for i in range(len(data)):
    #     img, mask = data[0][i], data[1][i]
    # print(mask[1])

    dataset_size = len(Data)
    print(f"Total number of images: {dataset_size}")
    valid_split = 0.2
    valid_size = int(valid_split*dataset_size)
    indices = torch.randperm(len(Data)).tolist()
    train_data = Subset(Data, indices[:-valid_size])
    val_data = Subset(Data, indices[-valid_size:])
    print(f"Total training images: {len(train_data)}")
    print(f"Total valid_images: {len(val_data)}")
    
    batch_size = 1
    counter = 0
    train_loader = DataLoader(train_data, batch_size = batch_size, 
                          num_workers = 0, 
                          pin_memory = 2,
                          shuffle = True)

    for j, data in tqdm(enumerate(train_loader), total = len(train_data) // batch_size):
        counter += 1
        img_tensor, mask_tensor = data[0], data[1]
        
    print('img_tensor size:', img_tensor.size())
    print('mask_tensor size:', mask_tensor.size())
    print(f'mask_tensor dtype, {mask_tensor.type()}')

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img_tensor.squeeze(0).permute(1, 2, 0))
    ax[1].imshow(mask_tensor.squeeze(0))
    plt.show()