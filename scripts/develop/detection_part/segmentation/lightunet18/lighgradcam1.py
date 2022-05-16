# 2022-03-24
# # Use the grad-cam to make layers output more expalinale
# For classification, the model predicts a list of the score category.
# For segmentation, the mode predicts these scores for every pixel in the image.
# The function is to compute the Class Activation Map (CAM).
# It isrequired to: 
# 1. Define a model wrapper to get the output tensor.
# 2. Define a target for sematic segmentation.

from turtle import forward
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn

import torch.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

from lightunet import LightUnet
from lightutils import load_model

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = 'datasets/S_kaggle_wildfire/000050.jpg'
root = 'havingfun/detection/segmentation/saved_imgs/'
modelparam_path = root + 'Lightunet18_MSE_Adam_1e4_e30.pth'

img_im = Image.open(img_path)
img_np = np.float32(np.array(img_im)) / 255
# img_cv2 = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
# img_np = np.float32(img_cv2) / 255
img_tensor = preprocess_image(img_np, 
                              mean = [0.485, 0.456, 0.406],
                              std = [0.229, 0.224, 0.225],)
print('======> input tensor size:', img_tensor.size())

checkpoint = torch.load(modelparam_path, map_location=torch.device(Device))
model = LightUnet().to(device = Device)
load_model(checkpoint, model)
model = model.eval()
img_tensor = img_tensor.to(device = Device)

pred_tensor = model(img_tensor) * 255
print('======> type of model putput:', type(pred_tensor))

# --------------------------------------------------------------------------------------------
# Some of the model outputs returning a dictionary with 'out' and 'aux' keys. the actual result needed is in 'out'.
#Therefore, for some model, it is needed to wrap the model first
class SegmentationModelOutputWrapper(nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']

# model = SegmentationModelOutputWrapper(model)
# output_tensor  = model(input_tensor)
# ---------------------------------------------------------------------------

normalized_mask = nn.functional.softmax(pred_tensor, dim = 1).cpu()
seg_classes = ['__Background__', 'Smoke', 'Flame',]
class2idx = {cls: idx for (idx, cls) in enumerate(seg_classes)}

smoke_category = class2idx['Smoke']
print('======> smoke_category:', smoke_category)
smoke_mask = normalized_mask[0, :, :, :].argmax(axis = 0).detach().cpu().numpy()
smoke_mask_uint8 = 255 * np.uint8(smoke_mask == smoke_category)
smoke_mask_gray = np.mean(smoke_mask == smoke_category)
smoke_mask_float = np.float32(smoke_mask == smoke_category)

mask_shaped = np.repeat(smoke_mask_uint8[:, :, None], 3, axis = 2)
print('======> shape of the shaped mask (np.array):', mask_shaped.shape)
both_imgs = np.hstack((img_im, np.repeat(smoke_mask_uint8[:, :, None], 3, axis = 2)))

# both_imgs = Image.fromarray(both_imgs)
# plt.imshow(both_imgs)
# plt.grid(False)
# plt.show()

# Take all the pixels that belong to "smoke" and sum their predictions
class SematicSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.mask = self.mask.to(device = Device)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# For the target, take all the pixels that belong to 'Smoke', and sum their 
target_layers = [model.neck[-1]]
targets = [SematicSegmentationTarget(smoke_category, smoke_mask_float)]
print(targets[0, :])
cam = GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available())
# grayscale_cam = cam(input_tensor=img_tensor,
#                     targets=targets)[0, :]
# cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
# cam_img = Image.fromarray(cam_img)
# plt.imshow(cam_img)
# plt.grid(False)
# plt.show()
