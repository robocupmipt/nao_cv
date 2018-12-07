#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Калибровка робота НАО ( Лариса) на 83 изображениях черно-белой шахматной доски формата A4
Сначала надо посадить робота прямо перед шахматной доской, которую надо приклеить к доске вертикальной.
Размер 1 клетки у доски 2.65 см. 7 декабря 4 доски хранятся в 112.

"""
from PIL import Image
from tqdm import tqdm
from naoqi import ALProxy
import numpy as np
import cv2 as cv
import os
os.chdir('/home/robocup/many_chessboards')#

GENERATE_IMAGES=True
if GENERATE_IMAGES:
        
    IP='192.168.1.5'
    PORT=9559
    resolution = 2    # VGA
    colorSpace = 11 
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
    motionProxy  = ALProxy("ALMotion", IP, PORT)
    postureProxy = ALProxy("ALRobotPosture", IP, PORT)
    for YAW in tqdm(np.arange(-0.5,0.5,0.05)):
       for PITCH in np.arange(-0.6,0.5,0.03):
           
           motionProxy.setAngles("HeadYaw",YAW,0.7)#
           motionProxy.setAngles("HeadPitch",PITCH,0.7)
           naoImage = Image.frombytes('RGB',(640,480),
            camProxy.getImageRemote(videoClient)[6])
           naoImage.save('IMAGE_'+str(YAW)+'_'+str(PITCH)+'.jpg')


image_names=os.listdir(os.getcwd())
CHESSBOARD_SIZE=(9,6)
CELL_SIZE = 2.65#

objectpoints_total=[]
imagepoints_total=[]
for image_name in tqdm(image_names):
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
print('camera_matrix')
print(camera_matrix)
print('rms')
print(rms)
print('dist_coefs')
print(dist_coefs)
'''
Лариса ; 83 изображения many_chessboards.zip
camera_matrix
[[  2.15481834e+03   0.00000000e+00   1.99388243e+02]
 [  0.00000000e+00   2.24961897e+03   3.40741433e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
rms
7.992188937971397

dist_coefs
array([[  5.22208878e+00,  -4.81402784e+01,   2.01759731e-01,
         -2.70434210e-01,   3.51670442e+02]])


Лариса 61 изображение  many_chessboards2.zip
camera_matrix
array([[  1.64708747e+03,   0.00000000e+00,   4.46115332e+02],
       [  0.00000000e+00,   1.14862198e+03,   2.54149736e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
rms
 8.777556383545917
 
dist_coefs
array([[  1.50501152e+00,   2.40128033e+00,   1.91165270e-01,
          3.68481189e-02,  -5.21361982e+01]])

Лариса 55 изображений many_chessboards3.zip ( положение самой доски то же, что и в предыдушем случае)

camera_matrix
 array([[  2.17745660e+03,   0.00000000e+00,   2.24045121e+02],
       [  0.00000000e+00,   8.72788401e+04,   7.21479440e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
rms
 8.717036910467252
dist_coefs
array([[  4.70863671e+00,  -7.59748568e+01,   1.92574245e-02,
         -1.88312012e-01,   7.81945326e+02]])

Лариса 106 изображений many_chessboards4.zip ( положение самой доски то же, что и в 2 предыдущих случаях, шаг по pitch 0.03 вместо 0.05 в остальных случаях
и значит, выборка должна быть более лучшего качества, чем предыдущие)
camera_matrix
[[  1.93987812e+03   0.00000000e+00   2.28895293e+02]
 [  0.00000000e+00   2.24652501e+03   3.33249335e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
rms
8.68592321783
dist_coefs
[[  4.33748198e+00  -5.66333361e+01   1.55668085e-01  -1.95557533e-01
    5.13039592e+02]]

Результаты первой и четвертой серии измерений ( самых крупных серий) очень хорошо согласуются друг с другом. Видимо, реальная матрица действительно какая-то такая.
'''
