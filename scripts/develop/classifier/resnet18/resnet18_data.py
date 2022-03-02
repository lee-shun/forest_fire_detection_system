#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: resnet18_data.py
#
#   @Author: Linhan qiao
#
#   @Date: 2021-11-11
#
#   @Email:
#
#   @Description:
#
# ------------------------------------------------------------------------------


import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# ratio of data to use for validation
valid_split = 0.2
# batch size
batch_size = 4
# path to the data root directory
root_dir = '/home/ls/dataset/fire_smoke_3_classes/trainused'

# define the training transforms and augmentations
train_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
valid_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# initial entire and test datasets
dataset = datasets.ImageFolder(root_dir, transform=train_transform)
dataset_test = datasets.ImageFolder(root_dir, transform=valid_transform)
print(f"Classes: {dataset.classes}")
dataset_size = len(dataset)
print(f"Total number of images: {dataset_size}")
valid_size = int(valid_split*dataset_size)
# training and validation sets
indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:-valid_size])
dataset_valid = Subset(dataset_test, indices[-valid_size:])
print(f"Total training images: {len(dataset_train)}")
print(f"Total valid_images: {len(dataset_valid)}")
# training and validation data loaders
train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_loader = DataLoader(
    dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4
)

