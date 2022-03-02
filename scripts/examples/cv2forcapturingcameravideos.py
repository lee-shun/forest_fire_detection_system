import cv2
import numpy as np # helpful in dealing with matrices
# read the images
img = cv2.imread("figurefastai.png")

# conver into gray scale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur the image
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # Blur kernel size (7,7), sigma x is 0
# edge detector
imgCanny = cv2.Canny(img, 100, 100) # thereshold 100 and 100
# imrove the thickness of the edges
kernel = np.ones((5, 5), np.uint8) # define the typ of objects as unit8
iterations = 1
imgDialation = cv2.dilate(imgCanny, kernel, iterations) # add the kernel, the iterations the kernels move
# eroded the image
imgEroded = cv2.erode(imgDialation, kernel, iterations)



# resize and crop the image
print(img.shape) # height, width, BGR. In cv2, x -> east, y -> south
imgResize = cv2.resize(img, (300, 200)) # define width then height

# cv2.imshow("Output", img)
# cv2.imshow("Gray image", imgGray)
# cv2.imshow("Blur image", imgBlur)
# cv2.imshow("Canny image", imgCanny)
# cv2.imshow("Dialation image", imgDialation)
cv2.imshow("Eroded image", imgEroded)
cv2.imshow("Resize image", imgResize)
cv2.waitKey(0)

# # video capure
# cap = cv2.VideoCapture("Smoke-sematic-segmentation/videoplayback.mp4")
# # desplay it, sequence of images
# while True:
#     success, img = cap.read()
#     cv2.imshow("video", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640) # wide 640, id number 3
# cap.set(4, 480) # height 480, id number 4
# cap.set(10, 100) # brightness 100, id number 10

# while True:
#     success, img = cap.read()
#     cv2.imshow("video", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

