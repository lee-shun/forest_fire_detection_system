from pickle import FALSE, TRUE
import torch

from tqdm import tqdm
from tqdm import trange
from time import sleep

import torch.nn as nn
import torch.optim as optim
from lightunet import LightUnet
from lightutils import (
    save_model,
    save_entire_model,
    load_model,
    seg_acc,
    check_accuracy,
    save_predictions_as_imgs,
    save_plots,
)
import argparse

# from albumentations.pytorch import ToTensorV2
import numpy as np
from lightdata import JinglingDataset, transform
from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 1
Num_workers = 0
Image_hight = 400
Image_weight = 400
Pin_memory = True
Valid_split = 0.2
Modeluse = LightUnet
# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type = int,
    default = 5,
    help = 'Numbers of epochs to train the network'
)
parser.add_argument(
    '-t',
    '--troot',
    type = str,
    default = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/images',
    help = 'Input the image dataset path'
)
parser.add_argument(
    '-m',
    '--mroot',
    type = str,
    default = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/masks',
    help = 'Input the mask dataset path'
)
parser.add_argument(
    '-l',
    '--lr',
    type = np.float32,
    default = 1e-4,
    help = 'Learning rate for training'
)
parser.add_argument(
    '-load',
    '--load',
    default = None,
    help = 'loading the trained model for prediction'
)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Img_dir = args['troot']
Mask_dir = args['mroot']
Learning_rate = args['lr']
Load_model = args['load']

# the device used fir training
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
print(model.eval())
print('#############################################################')
print(f'There are {total_params:,} total parameters in the model.\n')
# optimizer used for training
optimizer = optim.Adam(model.parameters(), lr = Learning_rate)
# loss function for training
loss_fn = nn.BCELoss()
# load dataset
data = JinglingDataset(img_dir = Img_dir,mask_dir = Mask_dir, transform = transform)
dataset_size = len(data)
print(f"Total number of images: {dataset_size}")
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(data)).tolist()
train_data = Subset(data, indices[:-valid_size])
val_data = Subset(data, indices[-valid_size:])
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(val_data)}")

print(f'\nComputation device: {Device}\n')

train_loader = DataLoader(train_data, batch_size = Batch_size, 
                          num_workers = Num_workers, 
                          pin_memory = Pin_memory,
                          shuffle = True)
val_loader = DataLoader(val_data, batch_size = Batch_size, 
                        num_workers = Num_workers, 
                        pin_memory = Pin_memory,
                        shuffle = True)

# def modeltrans(input, target):
#     preds = model(input)n_epochs, n_steps = 5, 100
#     if preds.shape != target.shape:
#         preds = TF.resize(preds, size=target.shape[2:])
#     return preds

def fit(train_loader, model, optimizer, loss_fn, scaler):
    print('====> Fitting process')

    train_running_loss = 0.0
    train_running_acc = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_data)):
        counter += 1

        img, mask = data
        img.to(device = Device)

        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask.to(device = Device)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)
            if preds.shape != mask.shape:
                preds = TF.resize(preds, size=mask.shape[2:])

            loss = loss_fn(preds, mask)
            train_running_loss += loss.item()
            train_running_acc += seg_acc(preds, mask).sum().item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

    n_epochs, n_steps = Num_epochs, len(train_data)
    with trange(1, n_epochs + 1, desc="All epochs") as epochs:
        for epoch in epochs:
            with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs)) as steps:
                for step in steps:
                    epochs.set_postfix(foo=epoch * n_steps + step)
                    steps.set_postfix(loss=loss.item(), baz=1 / step)
                    sleep(0.01)

        # # update tqdm loop
        # tqdm(enumerate(train_loader)).set_postfix(loss = loss.item())
        # tqdm(enumerate(train_loader)).set_postfix(loss = loss.item())

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * train_running_acc
    return epoch_loss, epoch_acc

def valid(val_loader, model, loss_fn):
    print('====> Validation process')

    val_running_loss = 0.0
    val_running_acc = 0.0
    counter = 0
    for i, data in tqdm(enumerate(val_loader), total = len(val_data)):
        counter += 1

        img, mask = data
        img.to(device = Device)

        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask.to(device = Device)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)
            if preds.shape != mask.shape:
                preds = TF.resize(preds, size = mask.shape[2:])

            val_loss = loss_fn(preds, mask)
            val_running_loss += val_loss.item()
            val_running_acc += seg_acc(preds, mask).sum().item()

    n_epochs, n_steps = Num_epochs, len(val_data)
    with trange(1, n_epochs + 1, desc="All epochs") as epochs:
        for epoch in epochs:
            with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs)) as steps:
                for step in steps:
                    epochs.set_postfix(foo=epoch * n_steps + step)
                    steps.set_postfix(loss=val_loss.item(), baz=1 / step)
                    sleep(0.01)

        # # update tqdm loop
        # tqdm(enumerate(train_loader)).set_postfix(loss = val_loss.item())

    val_epoch_loss = val_running_loss / counter
    val_epoch_acc = 100. * val_running_acc
    return val_epoch_loss, val_epoch_acc

def main():
    if Load_model is not None:
        pass
        # load_model(torch.load('Lightuent18S_1e5_e18.pth'), model)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # check_accuracy(val_loader, model, device = Device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Num_epochs):
        train_epoch_loss, train_epoch_acc = fit(train_loader, model,
                                                     optimizer, loss_fn, scaler)
        val_epoch_loss, val_epoch_acc = valid(val_loader, model,loss_fn)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

    # save entire model
    save_model(Num_epochs, model, optimizer, loss_fn)
    # check accuracy
    # check_accuracy(val_loader, model, device = Device)

    # print some examples to a folder
    # save_predictions_as_imgs(val_loader, 
    #                             model, 
    #                             folder = 'saved_imgs/', 
    #                             device = Device)

    # plot loss and acc
    save_plots(train_acc, val_acc, train_loss, val_loss)

    # # save final model
    save_entire_model(Num_epochs, model, optimizer, loss_fn)

    print('\n============> TEST PASS!!!\n')


if __name__ == "__main__":
    main()





