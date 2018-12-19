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
Различные матрицы и коэф для больших выборок. пока вопрос , что брать
Выборка 7 (верный подход к shape, консоль 24А)

camera_matrix
[[1.64865325e+03 0.00000000e+00 3.86372746e+02]
 [0.00000000e+00 9.75764487e+02 1.21598217e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
4.9776732877281145
dist_coefs
[[ 1.44204993e+00 -4.86278116e+00 -5.96676852e-03 -8.12728483e-02
   2.86299456e+01]]
Выборки 1,4(консоль 27А)

Directory
C:\Users\user\Downloads\many_chessboards14
camera_matrix
[[1.90647358e+03 0.00000000e+00 3.19745443e+02]
 [0.00000000e+00 1.80349894e+03 1.56042337e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
8.40679505434177
dist_coefs
[[ 2.77371885e+00 -2.86595154e+01 -8.44277083e-02 -3.24702463e-02
   7.26861870e+02]]


Выборка 6 (консоль 28А)

camera_matrix
[[1.13837380e+04 0.00000000e+00 3.41587189e+02]
 [0.00000000e+00 3.27591274e+03 2.21675525e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.1047639922229395
dist_coefs
[[ 2.40885209e+01 -1.41047909e+03  9.71122182e-01 -2.28036264e-02
  -2.70913949e+03]]

Выборка 5(консоль 29А)
SAMPLE SIZE 869
200951.77968478203
Directory
C:\Users\user\Downloads\many_chessboards5
camera_matrix
[[1.40493080e+04 0.00000000e+00 3.64520100e+02]
 [0.00000000e+00 4.62615377e+03 2.16972632e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.203838700866708
dist_coefs
[[ 4.85915005e+01 -6.22411466e+03  1.29388851e+00 -4.01371903e-02
  -4.28539001e+03]]
Выборки 1,2,3,4(консоль 30А)

Directory
C:\Users\user\Downloads\many_chessboards1234
camera_matrix
[[1.90536220e+03 0.00000000e+00 3.22329890e+02]
 [0.00000000e+00 1.85628461e+03 1.37908247e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
8.526984802369714
dist_coefs
[[ 2.54249405e+00 -1.60258922e+01 -1.14349634e-01 -4.89189274e-02
   5.32506387e+02]]



Выборки 1,2,3,4,5,6фильтрованная, MANYDIST(консоль 32A)


Выборки 1,4,MANYDIST(консоль 33А)
Предварительно на 516
camera_matrix
[[ 1.46240075e+03  0.00000000e+00  3.24806002e+02]
 [ 0.00000000e+00  1.00206827e+03 -6.17942773e+00]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]
rms
5.703961561913892
dist_coefs
[[ 1.22394222 -2.11858604 -0.21379144 -0.04868427  3.25447246]]

'''
