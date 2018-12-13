# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:58:26 2018

@author: user
"""
from tqdm import tqdm
import numpy as np
import cv2 as cv
import os
os.chdir('C:/Users/user/Downloads/many_chessboards3')

#os.chdir('C:/Users/user/Downloads/many_chessboards6')


image_names=os.listdir(os.getcwd())
CHESSBOARD_SIZE=(9,6)
CELL_SIZE = 2.65#CM

objectpoints_total=[]
imagepoints_total=[]
import time
t=time.time()
for image_name in tqdm(image_names[:140]):
   gray_image = cv.imread(image_name,0)
   object_points=[]
   (found,corners)=cv.findChessboardCorners(gray_image,CHESSBOARD_SIZE,flags =cv.CALIB_CB_ADAPTIVE_THRESH|cv.CALIB_CB_FILTER_QUADS)
   stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)#or 0.1
   image_points=[]
   point_indices=[]
   if found:
       point_indices = cv.cornerSubPix(gray_image, corners,
                (11,11), (-1,-1), stop_criteria)
   if len(point_indices)==CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1]:
       for i in range(len(point_indices)):
            image_points.append([point_indices[i][0][0],point_indices[i][0][1]])
            object_points.append([i%CHESSBOARD_SIZE[0], i//CHESSBOARD_SIZE[1],0])
       object_points=np.array(object_points).astype('float32')
       image_points=np.array(image_points).astype('float32')
       objectpoints_total.append(object_points)
       imagepoints_total.append(image_points)
   else:
       os.remove(image_name)
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(objectpoints_total, imagepoints_total,gray_image.shape,None,None)
print(time.time()-t)
print('Directory')
print(os.getcwd())
print('camera_matrix')
print(camera_matrix)
print('rms')
print(rms)
print('dist_coefs')
print(dist_coefs)
exs=[0.26,0.37,2.06,13.35,124]
points=[5,10,20,40,80]

from BallClassifier import HaarClassifier
'''
CELL_SIZE размер клетки равен 2.65 см
'''
camera_matrix = CELL_SIZE*camera_matrix
cls = HaarClassifier(classifier_dir = 'top_cascade.xml')
def get_real_coordinates(matrix, distortion, image):   
    image = cv.undistort(image, camera_matrix, dist_coefs)
    ball = cls.predict(image, save_image = False)
    real_coord = np.matmul(matrix, ball)
    return real_coord

#EXPONENTIAL REGRESSION
#0.24*1.08^X
'''
Диагональные коэфф-ты должны быть примерно равны друг другу
На 313 manually filtered
camera_matrix
[[1.09356907e+04 0.00000000e+00 2.19441556e+02]
 [0.00000000e+00 5.09510760e+03 1.55229412e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.04088690071522
dist_coefs
[[ 4.90942852e+01  2.97719108e+03  9.93898367e-01 -7.61275893e-01
  -8.19437050e+02]]
Без manually filtered
Directory
C:\Users\user\Downloads\many_chessboards6
camera_matrix
[[1.27072111e+04 0.00000000e+00 1.24021617e+02]
 [0.00000000e+00 5.94010906e+03 1.63659003e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.126758258967211
dist_coefs
[[ 5.73602739e+01  3.83814606e+03  1.12599721e+00 -1.30882996e+00
  -5.81097113e+03]]

Первое и четвертое Directory
C:\Users\user\Downloads\many_chessboards14
camera_matrix
[[1.56573705e+04 0.00000000e+00 2.83340306e+02]
 [0.00000000e+00 4.84352867e+03 1.93974114e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
8.664589313079357
dist_coefs
[[ 3.71310277e+01  8.26038759e+03  1.50376247e+00 -4.71517005e-02
  -2.50158224e+04]]
'''