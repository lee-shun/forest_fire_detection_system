import cv2
import numpy as np


# def save_image(image, addr, num):
#     address = addr + str(num) + '.jpg'
#     cv2.imwrite(address, image)

# videoCapture = cv2.VideoCapture('/home/qiao/dev/giao/dataset/imgs/M300test00/test1_infvideo/DJI_20211017104441_0001_T.MP4')

# success, frame = videoCapture.read()
# i = 0
# timeF = 10
# j = 0

# while success:
#     i = i + 1
#     if (i  % timeF == 0):
#         j = j +1
#         save_image(frame, './cutvideo/20220204/ir', j)
#         print('save image:', i)
#     success, frame = videoCapture.read()



cuttime = 28000
video_path = '/media/qiao/LS/DJI_20211017104441_0001_Z.MP4'
cover_path = '/home/qiao/dev/giao/dataset/imgs/M300test00/test1_infvideo/rgbcut028.jpg'

try:
    vc = cv2.VideoCapture(video_path)
    video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_hight = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vc.set(cv2.CAP_PROP_POS_MSEC, cuttime)
    rval, frame = vc.read()
    if rval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(cover_path, gray)
    else:
        print('fail reading')
except Exception as e:
    print(f'error: {e}')

    