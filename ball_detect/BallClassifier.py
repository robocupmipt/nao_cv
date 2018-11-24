# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:10:54 2018

@author: user
"""

import cv2
classifiers=['other_cascade.xml','top_cascade.xml','bottom_cascade.xml']
#First classifier was taken from https://github.com/dbloisi/detectball
#Other two classifiers were taken from http://www.dis.uniroma1.it/~labrococo/?q=node/459
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
    def predict(self,image,save_image=True, save_dir = 'DETECTED_BALL.jpg',
                warn_not_found=True):
        try:
            image1 = cv2.resize(image, 
                    (image.shape[0]//self.scale_factor,
                     image.shape[1]//self.scale_factor))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        except:
            print('No image received')
            raise Exception
        try:
            balls = self.haar_classifier.detectMultiScale(
                    image1, self.haar_params[0],self.haar_params[1])
            print(balls)
            balls = self.scale_factor*balls
        except:
            print('Exception while applying cascade')
            raise Exception
        for (x,y,w,h) in balls:
            image1 = cv2.rectangle(image1, (x,y),(x+w,y+h),(255,0,0),2)
        if save_image:
            cv2.imwrite(save_dir, image1)
        if len(balls)==0 and warn_not_found:
            print('No balls found - returning empty')
        return balls
cls = HaarClassifier(classifier_dir = classifiers[0])
img = cv2.imread('BALL1.jpg')
cls.predict(img, save_image=True, save_dir ='DETECTED_BALL1.jpg')  
img = cv2.imread('BALL2.jpg')
cls.predict(img, save_image=True, save_dir ='DETECTED_BALL2.jpg')  
  
