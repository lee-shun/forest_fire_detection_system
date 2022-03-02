# -*- coding: utf-8 -*-
# @Time    : 2019/4/28 13:52
# @Author  : ljf
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


# 1.1 tensor2Image Image格式进行绘图，展示
tensor1 = torch.randint(0,255,(300,300))
transform1 = transforms.ToPILImage(mode="L")
image1 = transform1(np.uint8(tensor1.numpy())) # Image接受的图像格式必须为uint8，否则就会报错
print(tensor1.size())
print(image1)
# image.show()
image1.save("gray.jpg")

# 1.2 Image2tensor tensor格式方便使用torch进行数据增强，也是模型训练的格式
# 先剪切，再转为tensor。底层也是PIL实现的
transform2 = transforms.Compose([transforms.RandomCrop([200,200],padding=10),transforms.ToTensor()])
image2 = Image.open("gray.jpg")
tensor2 = transform2(image2)
print(tensor2.size())

# 2.1 tensor2numpy 再1.1中也用到了，numpy格式主要用于容易转换数据格式，也有利于转为opencv格式。
array1 = tensor1.numpy()
print(array1.shape)
print(array1.dtype)

# 2.2 numpy2tensor 1.2有介绍，不再赘述
tensor3 = torch.Tensor(array1)
tensor4 = transforms.ToTensor()(array1)
print(tensor3.size())
print(tensor4.size()) # 会增加一个维度

# 3.1 numpy2opencv openc格式方便画目标框，图片上面写字(Image格式也可以实现，不是很熟悉，，，)
# opencv 读取出来就是numpy的数据格式
cv2.imshow("img",np.uint8(array1))
# cv2.waitKey()
# cv2.destroyAllWindows()

# 3.2 opencv2numpy
array2 = cv2.imread("./gray.jpg") # 这里使用opencv读取的是三通道，plt读取的是单通道。。暂时还没搞懂
print(array2.shape)
print(array2.dtype)

# 4.1 opecv2Image
image3 = Image.fromarray(array2,mode="RGB")
# image3.show()

# 4.2 Image2opencv
# 这里有两种方式，一种稍复杂点，但是可以保存数据形状
array3 = transforms.ToTensor()(image3).numpy()

# Image自带的属性，但是会打乱数据为一维
list1 = list(image3.getdata())
print(array3.shape)
print(list1)

