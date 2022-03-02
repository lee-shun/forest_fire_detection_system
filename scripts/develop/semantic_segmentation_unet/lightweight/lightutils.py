import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import os
root = os.path.dirname(os.path.join('/home/qiao/dev/giao/havingfun/detection/segmentation/saved_imgs/'))

modelname = 'Lightunet18'
lr = '1e4'
epochs = 'e5'
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
    print('====> Loading checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])

# compute accuracy
# segmentation codes
codes = ['Target', 'Void']
num_classes = 2
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']

def seg_acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim = 1)[mask]==target[mask]).float().mean()

def check_accuracy(loader, model, device = 'cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum())/(
                (preds + y).sum() + 1e-8
            )

    print(f'Got {num_correct}/{num_pixels} with acc: {num_correct/num_correct * 100:.2f}')
    print(f'Got dice score of: {dice_score/len(loader)}')
    model.train()

def save_predictions_as_imgs(loader, model, folder = root, device = 'cuda'):
    print('===========> saving prediction')
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{idx}.png')

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

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes +1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Ouput mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(root, show_imgs))

# if __name__ == '__main__':
    # save_model()
    # load_model()
    # save_model()
    # check_accuracy()
    # save_predictions_as_imgs()
    # save_plots()
    