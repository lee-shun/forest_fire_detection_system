from turtle import clear
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from lightunet import LightUnet
from lightutils import (
    save_model,
    save_entire_model,
    load_model,
    save_predictions_as_imgs,
    plot_img_and_mask,
    save_plots,
)

import argparse

# from albumentations.pytorch import ToTensorV2
import numpy as np
from lightdataPIL import JinglingDataset, Atransform
from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, 'havingfun/deving/blocks')
from  evaluateratios import Segratio

# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 2
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
    default = 10,
    help = 'Numbers of epochs to train the network'
)

parser.add_argument(
    '-l',
    '--lr',
    type = np.float32,
    default = 8.59e-2,
    help = 'Learning rate for training'
)

parser.add_argument(
    '-t',
    '--troot',
    type = str,
    default = 'datasets/S_google_wildfire',
    help = 'Input the image dataset path'
)
parser.add_argument(
    '-m',
    '--mroot',
    type = str,
    default = 'datasets/S_google_wildfire_label',
    help = 'Input the mask dataset path'
)

# classes add codes
codes = ['Background', 'Cloud', 'Smoke', 'Flame']
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']
print('name2id:', name2id)
num_classes = len(name2id)
print('num_classes:', num_classes)

# def acc_smoke(input, target):
#     target = target.squeeze(1)
#     mask = target != void_code
#     return (input.argmax(dim = 1)[mask] == target[mask]).float().mean()
# metric = acc_smoke
metric = Segratio(num_classes)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Img_dir = args['troot']
Mask_dir = args['mroot']
Learning_rate = args['lr']

# the device used fir training
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("=====>CUDA is available! Training on GPU...")
else:
    print("=====>CUDA is not available. Training on CPU...")

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Device = 'cpu'
# print(f'\nComputation device: {Device}\n')

# load the model
model = Modeluse(in_channels=3, out_channels=1)
model = model.to(device = Device)

# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
# print(model.eval())
print('#############################################################')
print(f'There are {total_params:,} total parameters in the model.\n')

# optimizer used for training
# optimizer = optim.Adam(model.parameters(), lr = Learning_rate)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# loss function for training
loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device = Device)

# load dataset
data = JinglingDataset(img_dir = Img_dir,mask_dir = Mask_dir, transform = Atransform)
dataset_size = len(data)
print(f"Total number of images: {dataset_size}")
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(data)).tolist()
train_data = Subset(data, indices[:-valid_size])
val_data = Subset(data, indices[-valid_size:])
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size = Batch_size, 
                          num_workers = Num_workers, 
                          pin_memory = Pin_memory,
                          shuffle = True)
val_loader = DataLoader(val_data, batch_size = Batch_size, 
                        num_workers = Num_workers, 
                        pin_memory = Pin_memory,
                        shuffle = True)

# resize tensor in up-sampling process                        
def sizechange(input_tensor, gate_tensor):
    sizechange = nn.UpsamplingBilinear2d(size = gate_tensor.shape[2:])
    out_tensor = sizechange(input_tensor)
    return out_tensor

# training process
def fit(train_loader, model, optimizer, loss_fn, scaler):
    print('====> Fitting process')

    train_running_loss = 0.0
    train_running_acc = 0.0
    train_running_mpa = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_data) // Batch_size):
        counter += 1
        img, mask = data
        img = img.to(device = Device)
        mask = mask.to(device = Device)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)
            # print('preds size before resize:', preds.size())
            # print('mask size:', mask.size())


            # for multiple class segmentation, the result should be 0, 1, 2, ...
            preds = torch.sigmoid(preds)
            # print('preds size after sigmoid:', preds.size())
            # preds = (preds > 0.5).float()

            # for now, the predictions are tensors
            # becaus of the U-net characteristic, the output is croped at edges
            # therefore, the tensor need to be resized
            # if preds.shape != mask.shape:
            #     preds = sizechange(preds, mask)
                # print('preds size after resize:', preds.size())

            # print(mask)
            # mask = mask.squeeze(1)
            loss = loss_fn(preds, mask)
            train_running_loss += loss.item()


            preds = preds.squeeze(1).permute(1, 2, 0)
            mask = mask.squeeze(1).permute(1, 2, 0)
            preds = (preds/255).cpu().detach().numpy().astype(np.uint8)
            mask = mask.cpu().detach().numpy().astype(np.uint8)

            # print('preds size:', preds.shape)
            # print('masks size:', mask.shape)

            hist = metric.addbatch(preds, mask)
            acc = metric.get_acc()
            train_running_acc += acc.item()

            mpa = metric.get_MPA()
            train_running_mpa += mpa.item()

        # backward
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        tqdm(enumerate(train_loader)).set_postfix(loss = loss.item(), acc = acc.item(), MPA = mpa.item())

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * train_running_acc / counter
    epoch_mpa = 100. * train_running_mpa / counter

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(preds)
    # ax[1].imshow(mask)
    # plt.show()
    return epoch_loss, epoch_acc, epoch_mpa

def valid(val_loader, model, loss_fn):
    print('====> Validation process')

    val_running_loss = 0.0
    val_running_acc = 0.0
    val_running_mpa = 0.0
    counter = 0
    for i, data in tqdm(enumerate(val_loader), total = len(val_data) // Batch_size):
        counter += 1

        img, mask = data
        img = img.to(device = Device)

        # mask = mask.unsqueeze(1)
        # mask = mask.float()
        mask = mask.to(device = Device)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)

            sig = nn.Sigmoid()
            preds = sig(preds)
            
            # if preds.shape != mask.shape:
            #     # preds = TF.resize(preds, size = mask.shape[2:])
            #     preds = sizechange(preds, mask)

            val_loss = loss_fn(preds, mask)
            val_running_loss += val_loss.item()

            preds = preds.squeeze(1).permute(1, 2, 0)
            mask = mask.squeeze(1).permute(1, 2, 0)
            preds = (preds/255).cpu().detach().numpy().astype(np.uint8)
            mask = mask.cpu().detach().numpy().astype(np.uint8)

            # print('preds size:', preds.shape)
            # print('masks size:', mask.shape)

            hist = metric.addbatch(preds, mask)
            val_acc = metric.get_acc()
            val_running_acc += val_acc.item()
            val_mpa = metric.get_MPA()
            val_running_mpa += val_mpa.item()

        tqdm(enumerate(val_loader)).set_postfix(loss = val_loss.item(), acc = val_acc.item(), mpa = val_mpa.item())

    val_epoch_loss = val_running_loss / counter
    val_epoch_acc = 100. * val_running_acc / counter
    val_epoch_mpa = 100. * val_running_mpa / counter
    return val_epoch_loss, val_epoch_acc, val_epoch_mpa

def main():

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # check_accuracy(val_loader, model, device = Device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Num_epochs):
        train_epoch_loss, train_epoch_acc, _ = fit(train_loader, model,
                                                    optimizer, loss_fn, scaler)
        # tqdm(enumerate(train_loader)).set_postfix(loss = train_epoch_loss(), acc = train_epoch_loss())
        val_epoch_loss, val_epoch_acc, _ = valid(val_loader, model,loss_fn)
        # tqdm(enumerate(val_loader)).set_postfix(loss = val_epoch_loss.item(), acc = val_epoch_loss())
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

        # save entire model
        save_model(Num_epochs, model, optimizer, loss_fn)

        # check accuracy
        # check_accuracy(val_loader, model, device = Device)


        save_plots(train_acc, val_acc, train_loss, val_loss)

        # # save final model
    save_entire_model(Num_epochs, model, optimizer, loss_fn)

    print('\n============> TEST PASS!!!\n')


if __name__ == "__main__":
    main()





