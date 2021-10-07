##[EAST MODEL](https://drive.google.com/file/d/14n9TxYnnYT26Q_-xAmWOkQkKfdjiP8bN/view)



import cv2
import pandas as pd
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

def east_detect(image,args): 

        (H, W) = image.shape[:2]

        (newW, newH) = (args["width"], args["height"])
        rW = W / float(newW)
        rH = H / float(newH)

        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        net = cv2.dnn.readNet(args["east"])

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (numRows, numCols) = scores.shape[2:4]
        angl = []

        for y in range(0, numRows):
            
            scoresData = scores[0, 0, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < args["min_confidence"]:
                    continue
                
                angle = anglesData[x]
                angl.append(angle*180/(np.pi))

        return np.median(angl)

def east(image_path,args):

        image = cv2.imread(image_path)
        angle = east_detect(image,args)
        #print("angle*********",angle)

        return image,angle

def hough_transforms(image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(gray,(11,11),0)
        edges = canny(thresh)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)

        return accum, angles, dists

def east_hough_line(image,args):
        image,angle = east(image,args)
        h, theta, d = hough_transforms(image)
        theta = np.rad2deg(np.pi/2-theta)
        #theta = np.rad2deg(theta-np.pi/2)
        margin = args['margin_tollerance']
        low_thresh = angle-margin
        high_thresh = angle+margin
        filter_theta = theta[theta>low_thresh]
        filter_theta = filter_theta[filter_theta < high_thresh]
        
        return image,np.median(filter_theta)

def rotate_bound(image, angle):
        
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def re_orient_east():

        args = {
            "image": "Image_path",
            "east": "/content/frozen_east_text_detection.pb",
            "min_confidence":0.5,
            "margin_tollerance":9,
            "width": 1280,
            "height": 1280
        }

        image,angle = east_hough_line(args['image'],args)

        if abs(angle) > 0.25:
            image = rotate_bound(image, angle)
            plt.imshow(image)

            cv2.imwrite("/content/corrected_image.png", image)

        print("Angle detectd is  {} ".format(angle))

        return image,angle

        
re_orient_east()
