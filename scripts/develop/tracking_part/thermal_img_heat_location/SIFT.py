import numpy as np
import imutils
import cv2
import time

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.isv2 = imutils.is_cv2()
    def stitch(self, images, ratio=0.75, reprojThresh = 4.0, showMatches = False):
        # detect keypoints and extract
        
        (imageB, imageA) = images
        start = time.time()
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        end = time.time()
        print('%.5f s' %(end - start))

        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        start = time.time()
        M = self.matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        end = time.time()
        print('%.5f s' %(end - start))

        # not enough matched keyponits to create a panorama
        if M is None:
            return None

        # otherwise, perspective warp to stitch images
        (matches, H, status) = M
        start = time.time()
        result = cv2.warpPerspective(imageA, H, 
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        end = time.time()
        print('%.5f s' %(end - start))

        # check if the keypoint matches should be visulaized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            end = time.time()
            print('%.5f s' %(end - start))
            return (result, vis)
        return result

    # recieving images, detect keypoints and extract local inveriance features
    # differecne of gaussian (DoG) keypoints detection and SIFT feature extraction
    def detectAndDescribe(self, images):
        # convert to grayscale
        # gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        gray = images
        if self.isv3:
            # detect and extract features from image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(images, None)

        # otherwise cv 2.4.x
        elif self.isv2:
            detector = cv2.FeatureDetector_create('SIFT')
            kps = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create('SIFT')
            (kps, features) = extractor.compute(gray, kps)

        # otherwise cv 4.5.x --on tp
        else:
            sift = cv2.SIFT_create()
            (kps, features) = sift.detectAndCompute(images, None)



        kps = np.float32([kp.pt for kp in kps])
        
        return (kps, features)

    # 4 params needed for match keypoints, keypoints and feature vectors of image I, keypoints and feature vectors of image II.
    # David Lowe's ratio nad RANSAC
    def matchKeyPoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create('BruteForce')
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homo
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None
    
    # draw the matches
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visulization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype = 'uint8')
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

if __name__ == '__main__':
    # load imagesand resize width of 400 (for afaster processing)
    imageA = cv2.imread('/home/qiao/dev/giao/dataset/imgs/M300test00/test1_infvideo/rgbcut069.jpg')
    imageB = cv2.imread('/home/qiao/dev/giao/dataset/imgs/M300test00/test1_infvideo/rgbcut028.jpg')

    imageA = imutils.resize(imageA, width = 400)
    imageB = imutils.resize(imageB, width = 400)

    # stitch and create panorama
    start = time.time()
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches = True)

    # show the images
    end = time.time()
    print('%.5f s' %(end - start))
    cv2.imwrite('vis_show028.jpg', vis)
    cv2.imwrite('result_show028.jpg', result)

