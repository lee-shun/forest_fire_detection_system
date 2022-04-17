import sklearn
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import os
root = os.path.dirname(os.path.join(
    'havingfun/detection/segmentation/saved_imgs/'
    ))

import numpy as np
import sklearn.metrics as metrics

modelname = 'Lightunet18_MSE_SGD'
lr = '8.59e2'
epochs = 'e10'
process_model_param = 'process_' + modelname + '_' + lr + '_' + epochs + '.pth'
model_param = modelname + '_' + lr + '_' + epochs + '.pth'
loss_imgs = 'Loss_'+ modelname + '_' + lr + '_' + epochs +'.png'
acc_imgs = 'Acc_' + modelname + '_' + lr + '_' + epochs +'.png'
show_imgs = 'Show_' + modelname + '_' + lr + '_' + epochs +'.png'

# save the model
def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(root,process_model_param))

def save_entire_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(root, model_param))

def load_model(checkpoint, model):
    print('======> Loading checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])

# compute accuracy
# segmentation codes
codes = ['Target', 'Void']
num_classes = 2
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']
       
def save_predictions_as_imgs(loader, model, folder = root, device = 'cuda'):
    print('===========> saving prediction')
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, 
            os.path.join(root, 'seg_result.png'),
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f'{folder}{idx}.png')

    model.train()

def save_plots(train_acc, val_acc, train_loss, val_loss):
    print(f'====> Saving processing ratios')
    plt.figure(figsize = (10, 7))
    plt.plot(
        train_acc, color = 'green', linestyle = '-', label = 'Train accuracy'
    )
    plt.plot(
        val_acc, color = 'blue', linestyle = '-', label = 'Validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(root, acc_imgs))

    plt.figure(figsize = (10, 7))
    plt.plot(
        train_loss, color = 'orange', linestyle = '-', label = 'Train loss'
    )
    plt.plot(
        val_loss, color = 'red', linestyle = '-', label = 'Validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Loss')
    plt.legend()
    
    plt.savefig(os.path.join(root, loss_imgs))

def plot_img_and_mask(img, pred, mask):
    print('=====> Saving prediction result')
    fig, ax = plt.subplots(3, 1)
    # plt.grid = False 
    # plt.xticks([]), plt.yticks([])

    fig = plt.figure()
    fig.set_size_inches(50,20)
    ax1 = fig.add_subplot(131)
    ax1.grid(False)
    ax1.set_title('Input Image')
    ax1.imshow(img)

    ax2 = fig.add_subplot(132)
    ax2.grid(False)
    ax2.set_title('Output Prediction')
    ax2.imshow(pred)
     
    ax3 = fig.add_subplot(133)
    ax3.grid(False)
    ax3.set_title('Target Mask')
    ax3.imshow(mask)

    plt.savefig(os.path.join(root, show_imgs))

# if __name__ == '__main__':
    # save_model()
    # load_model()
    # save_model()
    # check_accuracy()
    # save_predictions_as_imgs()
    # save_plots()
    