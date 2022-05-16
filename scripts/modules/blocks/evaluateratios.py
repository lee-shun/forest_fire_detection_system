# This block is for evaluate the segmentation result using confusion matrix and compute F1-score
# '''
# P/T  T   N
# T   TP  FP
# N   FN  TN
# '''
import torch
import numpy as np
import cv2
# from sklearn.metrics import confusion_matrix

__all__ = ['segmentationmatric']

class Segratio(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    # (TP + TN) / (TP + FP + FN + TN) 
    def get_acc(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc
    # TP / (TP + FP)
    def get_precision(self):
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return precision
    # mean pixel accuracy
    def get_MPA(self):
        precision = self.get_precision()
        mpa = np.nanmean(precision)
        return mpa
    # IoU: Intersection over union. intersection = TP union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    def iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis = 0) - np.diag(self.confusion_matrix)
        iou = intersection / union
        return iou
    # mean iou
    def meaniou(self):
        miou = np.nanmean(self.iou())
        return miou
    def get_connfusion_matrix(self, preds, masks):
        mask = (masks >= 0) & (masks < self.num_class)
        label = self.num_class * masks[mask] + preds[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    def frequency_weighted_IoU(self):
        # FWIOU = [(TP + FN) / (TP + FP + TN + FN)] * [TP / (TP + FP = FN)]
        freq = np.sum(self.confusion_matrix, axis = 1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0)
            - np.diag(self.confusion_matrix))
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    def addbatch(self, preds, masks):
        assert preds.shape == masks.shape
        self.confusion_matrix += self.get_connfusion_matrix(preds, masks)
        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

if __name__ == '__main__':
    # preds = cv2.imread('/home/qiao/dev/giao/dataset/imgs/jinglingseg1/images/img6.png')
    # masks = cv2.imread('/home/qiao/dev/giao/dataset/imgs/jinglingseg1/images/img9.png')
    # preds = np.array( preds / 255., dtype = np.uint8)
    # masks = np.array ( masks/ 255., dtype = np.uint8)

    preds = torch.randn(2, 1, 400, 400)
    masks = torch.randn(2, 1, 400, 400)
    preds = preds.squeeze(1).permute(1, 2, 0)
    masks = masks.squeeze(1).permute(1, 2, 0)
    print('preds size:', preds.shape)
    print('masks size:', masks.shape)

    preds = (preds/255).detach().numpy().astype(np.uint8)
    masks = masks.detach().numpy().astype(np.uint8)
    print('preds size:', preds.shape)
    print('masks size:', masks.shape)
    metric = Segratio(2)
    hist = metric.addbatch(preds, masks)
    pa = metric.get_acc()
    cpa = metric.get_precision()
    mpa = metric.get_MPA()
    iou = metric.iou()
    miou = metric.meaniou()
    print('preds size:', preds.shape)
    print('masks size:', masks.shape)
    print(1080*1920*3)
    print(hist, pa, cpa, mpa, iou, miou)

