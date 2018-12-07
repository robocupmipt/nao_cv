#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:20:09 2018

@author: robocup
"""

import numpy as np
import cv2 as cv
image_name='IMGEE2.jpg'
if image_name=='IMGEE2.jpg':
    CHESSBOARD_SIZE=(7,6)
    CELL_SIZE=2.4#sentimeters
elif image_name=='IMGEE.jpg':
    CHESSBOARD_SIZE=(6,5)
    CELL_SIZE=4.3#centimeters
gray_image = cv.imread(image_name,0)
#trying different chessboard size
#object_points = np.zeros((CHESSBOARD_SIZE[0],CHESSBOARD_SIZE[1],3), np.float32)
#object_points = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
object_points=[]
image_points=[]
(found,corners)=cv.findChessboardCorners(gray_image,CHESSBOARD_SIZE,flags =cv.CALIB_CB_ADAPTIVE_THRESH|cv.CALIB_CB_FILTER_QUADS)
stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                 30, 0.001)#or 0.1
point_indices = cv.cornerSubPix(gray_image, corners,
        (11,11), (-1,-1), stop_criteria)
for i in range(len(point_indices)):
    image_points.append([point_indices[i][0][0],point_indices[i][0][1]])
    object_points.append([i%CHESSBOARD_SIZE[0], i//CHESSBOARD_SIZE[1],0])
camera_matrix = np.zeros((3,3))
dist_coef = np.zeros(4)
object_points=np.array(object_points).astype('float32')
image_points=np.array(image_points).astype('float32')
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera([object_points], [image_points],gray_image.shape,None,None)



'''
ON BIG CHESSBOARD (IMGEE.jpg) CELL SIZE IS 4.3 SENTIMETERS
camera_matrix
array([[  1.43846455e+03,   0.00000000e+00,   2.53210045e+02],
       [  0.00000000e+00,   3.02522799e+03,   3.12894224e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

rms
7.77170501104008

dist_coefs
array([[  8.49542736e+01,  -4.53379925e+03,   3.84881761e-01,
         -2.02998630e+00,   1.22707172e+05]])

rvecs
[array([[ 0.21484913],
        [ 1.79561038],
        [ 2.5577579 ]])]

tvecs
[array([[  3.9699797 ],
        [  0.19620612],
        [ 31.33920988]])]
'''


'''
ON SMALL CHESSBOARD (IMGEE2.jpg) CELL SIZE IS 2.4 SENTIMETERS

camera_matrix
array([[ 690.89023932,    0.        ,  221.33058046],
       [   0.        ,  486.3910519 ,  241.28567818],
       [   0.        ,    0.        ,    1.        ]])

rms
9.167640319185624

dist_coefs
array([[ -2.52189534e-03,   6.23199558e+00,   7.35003441e-02,
         -3.69852072e-01,  -9.60081141e+00]])

rvecs
[array([[-0.06365906],
        [-0.3972909 ],
        [-1.58300872]])]

tvecs
[array([[  1.57999426],
        [  2.66029864],
        [ 10.01147979]])]
'''
