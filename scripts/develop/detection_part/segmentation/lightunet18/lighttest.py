import torch

from lightunet import LightUnet
from lightutils import (
    load_model,
    save_predictions_as_imgs,
    plot_img_and_mask,
)
import argparse

import torchvision

from PIL import Image
import matplotlib.pyplot as plt
import os
# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 1
Num_workers = 0
Image_hight = 400
Image_weight = 400
Pin_memory = True
Valid_split = 0.2
Modeluse = LightUnet
root = 'havingfun/detection/segmentation/saved_imgs/'
modelparam_path = root + 'Lightunet18_MSE_Adam_1e5_e30.pth'

# the device used fir training
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(modelparam_path, map_location=torch.device(Device))

# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()

# specifying whether to test the trained model
parser.add_argument(
    '-tar',
    '--tar_img',
    type = str,
    default = 'datasets/S_kaggle_wildfire/000005.jpg',
    help = 'Load the target image to be detected'
)
tarmask_path = 'datasets/S_kaggle_wildfire_label/label_000005.jpg'

args = vars(parser.parse_args())
Target_img = args['tar_img']



# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
# print(model.eval())

print(f'==============> There are {total_params:,} total parameters of this model.\n')

def main():
    
    os.system('export DISPLAY=:0.0')
    
    img_path = Target_img
    img_im = Image.open(img_path).convert('RGB')
    mask_im =Image.open(tarmask_path).convert('L')
    trans2tensor = torchvision.transforms.ToTensor()
    img_tensor = trans2tensor(img_im).unsqueeze(0).to(device = Device)
    load_model(checkpoint, model)
    pred_tensor = 255 * model(img_tensor)

    pred_tensor = pred_tensor.squeeze(1)
    trans2img = torchvision.transforms.ToPILImage()
    pred_im = trans2img(pred_tensor).convert('L')
    plt.imshow(pred_im)
    plt.grid(False)
    plt.show()
    plot_img_and_mask(img_im, pred_im, mask_im)

    # print some examples to a folder
    # save_predictions_as_imgs(val_loader, 
    #                             model, 
    #                             folder = 'havingfun/detection/segmentation/saved_imgs', 
    #                             device = Device)

    # plot loss and acc

if __name__ == '__main__':
    main()