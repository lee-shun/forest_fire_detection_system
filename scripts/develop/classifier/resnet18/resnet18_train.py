#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: resnet34_train.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2021-11-11
#
#   @Email:
#
#   @Description:
#
# ------------------------------------------------------------------------------

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from resnet18_model import Resnet18
from resnet18_utils import save_model, save_entire_model, save_plots
from resnet18_data import train_loader, valid_loader, dataset
from tqdm.auto import tqdm

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=60,  # training epochs
    help='number of epochs to train our network for')
args = vars(parser.parse_args())

# learning rate
lr = 1e-5
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
# build the model
# model = Resnet34(img_channels=3, num_classes=3)
model = Resnet18(img_channels=3, num_classes=3)
model = model.to(device)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters()
                             if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.CrossEntropyLoss()


def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# validation
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    # we need two lists to keep track of class-wise accuracy
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # calculate the accuracy for each class
            correct = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                # class_correct[label] += correct.item()
                class_total[label] += 1

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    # print the accuracy for each class after every epoch
    print('\n')
    for i in range(len(class_names)):
        print(
            f"Accuracy of class {class_names[i]}: {100*class_correct[i]/class_total[i]}"
        )
    print('\n')

    return epoch_loss, epoch_acc


if __name__ == '__main__':

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion,
                                                     dataset.classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print('saving model params')
        save_model(epochs, model, optimizer, criterion)
        print('-' * 50)

    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    save_entire_model(epochs, model, optimizer, criterion)
    print('TRAINING COMPLETE')
