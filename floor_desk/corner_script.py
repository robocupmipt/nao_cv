# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:19:45 2019

@author: user
"""

import os
os.chdir('C:\\Users\\user\\Pictures')
import cv2
image0 = cv2.imread('IMAGE_0_0_35.jpg')
image1 = cv2.imread('IMAGE_0_0_1_35.jpg')
image2 = cv2.imread('IMAGE_0_0_2_35.jpg')
image3 = cv2.imread('IMAGE_0_0_3_35.jpg')


_,corners_0 = cv2.findChessboardCorners(image0,(7,7))
_,corners_1 = cv2.findChessboardCorners(image1,(7,7))
_,corners_2 = cv2.findChessboardCorners(image2,(7,7))
_,corners_3 = cv2.findChessboardCorners(image3,(5,7))
print('Corners from image 0')
print(corners_0)
print('Corners from image 1')
print(corners_1)
print('Corners from image 2')
print(corners_2)
print('Corners from image 3')
print(corners_3)


