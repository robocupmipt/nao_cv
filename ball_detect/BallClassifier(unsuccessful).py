# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:07:33 2018

@author: user
"""
'''
В каждом круге как можно больше должно быть белым либо черным.
Т.е мы пиксель внутри каждого из кругов сравниваем с белым ( или черным) и отклонение от белого/черного записываем
'''
import random
import math
import os
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm
class Detector():
    def __init__(self, canny_thresholds = (100,200),
                         hough_dp=1, hough_params=(50,30),
                         hough_mindist=50,radius_minmax = (1,70),
                         filter_lowhigh = (25,225), filter_numsamples=10000):
        if radius_minmax[0]>radius_minmax[1]:
            radius_minmax = radius_minmax[::-1]
        if filter_lowhigh[0]>filter_lowhigh[1]:
            filter_lowhigh=filter_lowhigh[::-1]
        if canny_thresholds[0]>canny_thresholds[1]:
            canny_thresholds = canny_thresholds[::-1]
        self.CANNY_THRESHOLDS = canny_thresholds
        self.HOUGH_DP = hough_dp
        self.HOUGH_PARAMS = hough_params
        self.HOUGH_MINDIST = hough_mindist
        self.RADIUS_MINMAX = radius_minmax
        self.FILTER_LOWHIGH = filter_lowhigh
        self.FILTER_NUMSAMPLES = filter_numsamples
    def get_score(self,circle):#REPLACE RANDOMIZED ALGORITHM WITH PURE ITERATING
        #print(circle)
        x_circle, y_circle, r_circle = circle
        wrong_count=1#to avoid 0/0 division
        low_count=0
        high_count=0
        LOW_THRESHOLD,HIGH_THRESHOLD = self.FILTER_LOWHIGH[0], self.FILTER_LOWHIGH[1]
        #for i in range(self.FILTER_NUMSAMPLES):
        #    alpha = 2 * math.pi * random.random()
        # random radius
         #   r = r_circle* math.sqrt(random.random())
        # calculating coordinates
        #    x = r* math.cos(alpha) + x_circle
         #   y = r * math.sin(alpha) + y_circle
        for x in range(x_circle - r_circle, x_circle+r_circle):
            for y in range(y_circle - r_circle, y_circle+r_circle):
                if ((x_circle-x)**2+(y_circle-y)**2<r_circle**2
                    and x>0 and x<self.img_s.shape[0]
                    and y>0 and y<self.img_s.shape[1]):
                    pixel = self.img_s[int(x),int(y)]
                    if (pixel<LOW_THRESHOLD):
                        low_count+=1
                    elif pixel>HIGH_THRESHOLD:
                        high_count+=1
                    else:
                        wrong_count+=1
        #print('hhh')
        return (low_count+high_count)/(low_count+high_count+wrong_count)#, (low_count/(high_count+1))
    def predict(self,image_dir, save = True,
        show = False,save_dir = 'DETECTED_BALL.jpg'):
        #print(image_dir)
        img = cv2.imread(image_dir)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.img_s = img_hsv[:,:,1]#Saturation of image
        
        edges = cv2.Canny(img, self.CANNY_THRESHOLDS[0], self.CANNY_THRESHOLDS[1])
        
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp = self.HOUGH_DP,
        minDist = self.HOUGH_MINDIST, 
        param1 = self.HOUGH_PARAMS[0],param2 = self.HOUGH_PARAMS[1],
        minRadius = self.RADIUS_MINMAX[0],maxRadius = self.RADIUS_MINMAX[1])
        circles = np.uint16(np.around(circles))
        print(circles)
        scores = [self.get_score(circle) for circle in circles[0,:]]
#        ratios =  [self.get_score(circle)[1] for circle in circles[0,:]]
        circle_index = np.argmax(scores)
        
        answ_circle = circles[0,:][circle_index]
        for circle,score in zip(circles[0,:],scores):
            print((circle,score))
        cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),2)
        cv2.circle(img,(circle[0],circle[1]),2,(0,0,255),3)
        if save:
            cv2.imwrite(save_dir,img)
        if show:
            cv2.imshow(img)
        return answ_circle
det = Detector(filter_lowhigh=(20,235))
det.predict('BALL1.jpg',save_dir = 'DETECTED_BALL1.jpg')
det.predict('BALL2.jpg',save_dir = 'DETECTED_BALL2.jpg')

      
