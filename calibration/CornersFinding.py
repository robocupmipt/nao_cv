#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Калибровка робота НАО ( Лариса) на 83 изображениях черно-белой шахматной доски формата A4
"""
from PIL import Image

from naoqi import ALProxy
import numpy as np
import cv2 as cv
import os
os.chdir('/home/robocup/many_chessboards')#

GENERATE_IMAGES=False
if GENERATE_IMAGES:
    print('Do you really want to GENERATE NEW IMAGES and OVERWRITE EXISTING ONES? PRESS Y is Yes')
    if input()[0].lower()=='y':
        
        IP='192.168.1.5'
        PORT=9559
        resolution = 2    # VGA
        colorSpace = 11 
        camProxy = ALProxy("ALVideoDevice", IP, PORT)
        videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
        motionProxy  = ALProxy("ALMotion", IP, PORT)
        postureProxy = ALProxy("ALRobotPosture", IP, PORT)
        YAW=-0.5
        PITCH=-0.5
        for YAW in np.arange(-0.5,0.5,0.05):
           print(YAW)
           for PITCH in np.arange(-0.5,0.5,0.05):
               
               motionProxy.setAngles("HeadYaw",YAW,0.3)#
               motionProxy.setAngles("HeadPitch",PITCH,0.3)
               naoImage = Image.frombytes('RGB',(640,480),
                camProxy.getImageRemote(videoClient)[6])
               naoImage.save('IMAGE_'+str(YAW)+'_'+str(PITCH)+'.jpg')


image_names=os.listdir(os.getcwd())
CHESSBOARD_SIZE=(9,6)#To define precisely further
CELL_SIZE = 2.65#To define precisely further

from tqdm import tqdm
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

'''
Размер 1 клетки 2.65 см. 7 декабря 4 таких доски хранятся в 112, еще одна хранится у Дмитрия Карпова в качестве эталона.
camera_matrix
[[  2.15481834e+03   0.00000000e+00   1.99388243e+02]
 [  0.00000000e+00   2.24961897e+03   3.40741433e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
rms
7.992188937971397

dist_coefs
array([[  5.22208878e+00,  -4.81402784e+01,   2.01759731e-01,
         -2.70434210e-01,   3.51670442e+02]])

'''