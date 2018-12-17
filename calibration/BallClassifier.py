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
camera_matrix = np.zeros((3,3))
camera_matrix[0,0]=1462
camera_matrix[0,2]=3248
camera_matrix[1,1]=1002
camera_matrix[1,2]=-6
camera_matrix[2,2]=1
dist_coeffs = np.array([1.223,-2.118,-0.213,-0.048,3.254])
rms = 5.70

cell_size=2.65#сантиметров
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
Матрицу камеры с учетом дисторсии пока не корректируем, т.к ее коррекция пока дает ошибочный результат
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
        dist_coeffs=dist_coeffs,include_distortion=False):
        if include_distortion:
            processed_matrix = None
            processed_image=None
            print('Distortion coeffs are not still handled correctly')
            raise Exception
        else:
            processed_matrix = camera_matrix
            processed_invmatrix = np.linalg.inv(processed_matrix)
            processed_image = image        
        image_coords = self.predict(
        processed_image, save_image = False,print_=False)
        if image_coords is None:
            return []
        x, y, w, h = image_coords[0]
        image_coords = [[x,y,1], [x, y+h,1],[x+w,y,1],[x+w,y+h,1]]
        real_coords = [cell_size*np.matmul(processed_invmatrix, np.array(ball_coord))
            for ball_coord in image_coords]
        return real_coords
        
cls = HaarClassifier(classifier_dir = classifiers[0])
image = cv2.imread('BALL1.jpg')
cls.predict(image, save_image=True, save_dir ='DETECTED_BALL1.jpg')  
image = cv2.imread('BALL2.jpg')
cls.predict(image, save_image=True, save_dir ='DETECTED_BALL2.jpg')  



        
    
    