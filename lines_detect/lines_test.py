#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:03:09 2018

@author: robocup
"""
USER_DIR = '/home/robocup/nao_vision/lines_detect/'
###Add a
import os
os.chdir(USER_DIR)
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

img1 = cv2.imread('lines1.jpg')
img2 = cv2.imread('lines2.jpg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

image = mpimg.imread('solidWhiteCurve.jpg')

cropped_image = region_of_interest(
    image,
    np.array([region_of_interest_vertices], np.int32),
)
plt.figure()
plt.imshow(cropped_image)

# Convert to grayscale here.
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

# Call Canny Edge Detection here.
cannyed_image = cv2.Canny(gray_image, 100, 200)
# Moved the cropping operation to the end of the pipeline.
cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)

plt.figure()
plt.imshow(cropped_image)
plt.figure()
plt.imshow(cannyed_image)

plt.show()

#
#
#
#gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#kernel_size = 5
#blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
#
#edges = cv2.Canny(blur_gray, 100, 200)
#
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
#
#for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#plt.imshow( edges)
##plt.imshow(img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()