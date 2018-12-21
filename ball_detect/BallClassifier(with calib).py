# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:10:54 2018

@author: user
"""

import random
import math
import os
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm
os.chdir('C://Users//user//Downloads')
classifiers=['ball_cascade.xml','top_cascade.xml','bottom_cascade.xml']

camera_matrix=np.array([[ 1.47612124e+03,  0.00000000e+00,  3.09005307e+02],
 [ 0.00000000e+00,  1.06518129e+03, -6.29354626e+01],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

dist_coefs=np.array([[ 1.05995926, -1.21305323, -0.25346465, -0.04999029,  1.28114937]])
cell_size=2.65#сантиметров


#img = cv2.imread('BALL1.jpg')
#
#h,  w = img.shape[:2]
#mtx=np.array(camera_matrix)
#dist=np.array(dist_coeffs)
#w,h=480,640
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#
#dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
## crop the image
#x,y,w,h = roi
#dst1 = dst1[y:y+h, x:x+w]
#cv2.imwrite('calibresult1.png',dst1)
#
#
#mean_error = 0
#for i in xrange(len(objpoints)):
#    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#    tot_error += error


#mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#assert dst.max()>0
## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
## undistort
#dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
#assert dst.max()>0
## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)
'''
Значения матрицы камеры и коэффициентов дисторсии пока неточные,
они будут обновлены позднее по завершению расчетов.
Размер клетки измеряется в сантиметрах, как и все остальные реальные координаты.
Измените cell_size, если хотите это поменять

https://hub.packtpub.com/camera-calibration/
s*(x y 1) = (CAMERA MATRIX)*(1 0 0 0
                             0 1 0 0
                             0 0 1 0) *(X Y Z 1)T
где у первого множителя во второй части размерность 3*3 у второго 3*4 у третьего 1*4
если референсная плоскость находится в центре камеры
Матрицу внешней калибровки пока считаем единичной
'''

class HaarClassifier():
    def __init__(self,classifier_dir = 'ball_cascade.xml',
                 haar_params=(1.3,5),scale_factor = 1):
        '''
        integer scale factor > 3 (4 or above) didn't lead to correct detection
        Average prediction time with scale factor 3 (on my computer) is 0.03 sec
        '''
        self.haar_classifier = cv2.CascadeClassifier(classifier_dir)
        self.haar_params = haar_params
        self.scale_factor = scale_factor
    def predict_onimage(self,image,save_image=True, save_dir = 'DETECTED_BALL.jpg',print_=True):
        try:
            image1 = cv2.resize(image, 
                    (image.shape[0]//self.scale_factor,
                     image.shape[1]//self.scale_factor))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        except:
            if print_:
                print('No image received')
                raise Exception
            return None
        try:
            balls = self.haar_classifier.detectMultiScale(
                    image1, self.haar_params[0],self.haar_params[1])
            print(balls)
            balls = self.scale_factor*balls
        except:
            if print_:
                print('Exception while applying cascade')
                raise Exception
            return None
        for (x,y,w,h) in balls:
            image1 = cv2.rectangle(image1, (x,y),(x+w,y+h),(255,0,0),2)
        if save_image:
            cv2.imwrite(save_dir, image1)
        if len(balls)==0:
            if print_:
                print('No balls found - returning empty')
                raise Exception
            return None
        return balls
    def predict(self, image, camera_matrix=camera_matrix, 
        dist_coeffs=dist_coeffs,include_distortion=True,save_image=True,save_dir='DETECTED_BALL.jpg'):
        if include_distortion:
            w,h=image.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w,h),1,(w,h))

            processed_image = cv2.undistort(image, camera_matrix,dist_coeffs, None, newcameramtx)

# crop the image
            x,y,w,h = roi
            processed_image=processed_image[y:y+h, x:x+w]
            processed_matrix = newcameramtx

        else:
            processed_matrix = camera_matrix
            processed_image = image  
        processed_invmatrix = np.linalg.inv(processed_matrix)
        processed_involdmatrix = np.linalg.inv(camera_matrix)
        image_coords = self.predict_onimage(
        processed_image, save_image = False,print_=False)
        if image_coords is None:
            return []
        x, y, w, h = image_coords[0]
        image_coords = [[x,y,1], [x, y+h,1],[x+w,y,1],[x+w,y+h,1]]
        real_coords = [cell_size*np.matmul(processed_involdmatrix, np.array(ball_coord))
            for ball_coord in image_coords]
        real_xywh =( real_coords[0][0],real_coords[0][1],
        real_coords[3][0]-real_coords[0][0],real_coords[3][1]-real_coords[0][1])
        return real_coords
        
cls = HaarClassifier(classifier_dir = classifiers[0])
image = cv2.imread('BALL1.jpg')
cls.predict_onimage(image, save_image=True, save_dir ='DETECTED_BALL1.jpg')  
image = cv2.imread('BALL2.jpg')
cls.predict(image, save_image=True, save_dir ='DETECTED_BALL2.jpg')  



        
    
    
