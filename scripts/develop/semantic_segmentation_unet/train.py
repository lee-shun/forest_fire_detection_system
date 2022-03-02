import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 255  # 1280 originally
IMAGE_WIDTH = 255  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "/home/ls/dataset/Unet_Smoke_segmentation/images"
TRAIN_MASK_DIR = "/home/ls/dataset/Unet_Smoke_segmentation/binary_mask"
VAL_IMG_DIR = "/home/ls/dataset/chimney_Somke_segmentation/train/images"
VAL_MASK_DIR = "/home/ls/dataset/chimney_Somke_segmentation/train/masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    num_correct = 0
    num_pixels = 0
    dice_score = 0

    loss_in_loop = []
    acc_in_loop = []
    dice_in_loop = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            preds = torch.sigmoid(model(data))
            preds = (preds > 0.5).float()
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * targets).sum()) / (
                (preds + targets).sum() + 1e-8)

        loss_in_loop.append(loss.item())
        acc_in_loop.append((num_correct / num_pixels * 100).item())
        dice_in_loop.append(dice_score.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        # print(loss_in_loop)
        # print(acc_in_loop)
        # print(dice_in_loop)

    return np.mean(loss_in_loop), np.mean(acc_in_loop), np.mean(dice_in_loop)


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(),
        A.Normalize(),
        ToTensorV2(),
    ], )

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ], )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, loss_fn, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    tra_loss_list = []
    tra_acc_list = []
    tra_dice_list = []

    val_loss_list = []
    val_acc_list = []
    val_dice_list = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_dice = train_fn(train_loader, model,
                                                     optimizer, loss_fn,
                                                     scaler)
        tra_loss_list.append(train_loss)
        tra_acc_list.append(train_acc)
        tra_dice_list.append(train_dice)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        val_loss, val_acc, val_dice = check_accuracy(val_loader,
                                                     model,
                                                     loss_fn,
                                                     device=DEVICE)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_dice_list.append(val_dice)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )

    torch.save(model.state_dict(), './ModelParams/final.pth')
    print("save the model as final.pth!\n")

    np.savetxt("tra_loss_list.csv", tra_loss_list, delimiter=",")
    np.savetxt("tra_acc_list.csv", tra_acc_list, delimiter=",")
    np.savetxt("tra_dice_list.csv", tra_dice_list, delimiter=",")

    np.savetxt("val_loss_list.csv", val_loss_list, delimiter=",")
    np.savetxt("val_acc_list.csv", val_acc_list, delimiter=",")
    np.savetxt("val_dice_list.csv", val_dice_list, delimiter=",")


if __name__ == "__main__":
    main()
