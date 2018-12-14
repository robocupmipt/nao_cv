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
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(objectpoints_total, imagepoints_total,gray_image.shape[::-1],None,None)
print('camera_matrix')
print(camera_matrix)
print('rms')
print(rms)
print('dist_coefs')
print(dist_coefs)

