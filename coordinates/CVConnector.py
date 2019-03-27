# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:02:08 2019

@author: asus
"""


import time
import cv2
from PIL import Image, ImageDraw
import numpy as np
import time
import qi



class CVConnector(object):
    def __init__(self, ip, 
                 ballfinder_dirs=['top_cascade.xml','other_cascade.xml','bottom_cascade.xml'],
                 camera_id=1):
        '''
        ballfinder_dir - directory of classifier
        camera_id - id of camera.
        Note: we use top_cascade.xml only with camera_id = 0
        and bottom_casca
        '''
        self.IP = ip
        self.PORT = 9559
        self.ballfinders= [cv2.CascadeClassifier(ballfinder_dir)
        for ballfinder_dir in ballfinder_dirs]
        self.camera_id = camera_id
        self.last_image=None
        self.image1=None
    def get_image(self, image_dir=None, yaw=0,pitch=0,set_speed=0.1):
        '''
        yaw is a horisontal angle ( -2.0857 to 2.0857 )
        pitch is a vertical angle ( -0.6720 to 0.5149)
        speed is a speed of head rotation ( from 0 to 1)
        num_balls = number of balls
        Note that max abs of pitch value depends on the max abs of yaw value
        http://doc.aldebaran.com/1-14/family/robots/joints_robot.html
        '''
        ses = qi.Session()
        ses.connect(self.IP)
        video = ses.service('ALVideoDevice')
        motionProxy = ses.service('ALMotion')
        motionProxy.setAngles(["HeadPitch","HeadYaw"],[pitch,yaw],set_speed)#

        if image_dir:
            self.last_image = cv2.imread(image_dir)
            self.last_shape=self.last_image.shape
        else:
            videoClient = video.subscribeCamera("python_client",
                                                self.camera_id, 2, 11, 5)
        
            naoImage = video.getImageRemote(videoClient)
            video.unsubscribe(videoClient)
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            im = Image.frombytes("RGB", (imageWidth, imageHeight), str(array))
            if im is None:
                print('IMAGE TAKING FAILED')
            self.last_image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            self.last_shape=self.last_image.shape
    def _get_ball(self, get_image=True,
                  scale_factor=1,
                  haar_params=(1.3, 5),
                  save_image=False,
                  save_dir='DETECTED_BALL.jpg',
                  print_=False,
                  num_balls=1):
        '''
        if get_image=True, we make new image, with parameters yaw,pitch,speed
        scale_factor is a parameter on which we rescale image before detect
        haar_params are params for ball finder
        if save_image=True, we save image after detection in save_dir.
        '''
        if get_image or (self.last_image is None):
            self.get_image()
        print('GETTING')

        print('RESIZING')
        if scale_factor!=1:
            image1 = cv2.resize(self.last_image, 
                    (self.last_image.shape[0] // scale_factor,
                     self.last_image.shape[1] // scale_factor))
        else:
            image1=self.last_image
        assert image1 is not None
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        print(image1.shape)
        assert image1 is not None
        self.image1 = image1

        balls=[]
        for ballfinder in self.ballfinders:
            if len(balls)==0:
                balls_found = scale_factor*ballfinder.detectMultiScale(
                    image1, haar_params[0],haar_params[1])
                print(balls_found)
                balls = balls_found
        print(balls)
        balls = balls[:num_balls]
        for (x, y, w, h) in balls:
            image1 = cv2.rectangle(image1, (x,y),(x+w,y+h),(255,0,0),2)
            if image1 is not None:
                self.last_shape=image1.shape
        if save_image:
            cv2.imwrite(save_dir, image1)

        if len(balls)==0:
            if print_:
                print('No balls found - returning empty')
                #raise Exception
            return None
        
        return balls[0]

    def get_all_ball_data(self):
        ball_coords = self._get_ball()
        if ball_coords is None or len(ball_coords) == 0:
            #cv2.imwrite("images/bad/{}.jpg".format(time.time()), self.last_image)
            return None, None, None, None


        x, y, w, h = ball_coords
        img = self.last_image
        #cv2.rectangle(img,
        #              (x, y), (x + w, y + h),
        #              (255, 0, 0), 2)
        image_h, image_w = self.last_shape[:2]

        #cv2.imwrite("images/good/{}.jpg".format(time.time()), img)

        print("DEBUG: " + str(ball_coords))
        print("DEBUF: " + str(img.shape[:2]))

        y = image_h - y
        center_x = x + w // 2 - image_w // 2
        center_y = y + h // 2 - image_h // 2
        return center_x, center_y, w, h
        

import numpy as np
import math

def position_calc(k1, k2, x0, y0):
    """
    Makes necessary calculations
    
    Arguments:
        k1 - rotation around Z axis angle. from -1 to 1
        k2 - incline  angle.               from  0 t0 1
        x0 - x coordinate of the point on the screen, pixels
        y0 - y coordinate of the point on the screen, pixels
        
    """   
    
    pi  = np.pi
    cos = math.cos
    sin = math.sin


    ###__constants__####

    h0 =  45.959                    #height of the neck joint, cm
    h2 =  5.071                     #position of the photomatrix relative to the neck joint along the Y axis, cm
    h3 =  6.364                     #position of the photomatrix relative to the neck joint along the Z axis, cm
    f  =  600                       #photomatrix-lens distanse, px


    e1 =  np.array([0, 0, 1])              #Z axis vector 
    e2 =  np.array([1, 0, 0])              #X axis vector
    H  =  h0 * e1                          #neck joint vector
    L0 =  np.array([0, h2, h3])            #position of the lens reletive to neck joint vector
    q0 =  -np.array([x0, f, y0])            #position of the point on the photomatrix vector reletive to lens
    Q0 =  L0 + q0                          #position of the point on the photomatrix vector reletive to neck joint



    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


    a1 =  pi/2 * k1                  #rotation around Z axis angle
    a2 = -pi/2 * k2                  #incline  angle

    R1 = rotation_matrix(e1, a1)     #rotaion around Z axis matrix
    R2 = rotation_matrix(e2, a2)     #incline up-down matrix
    R  = np.matmul(R1, R2)           #full rotation matrix


    Q = R.dot(Q0) + H                #position of the point on the screen reletive to floor vector
    q = R.dot(q0)                    #ligth ray vector
    
    return Q, q






IP = '192.168.1.61'
sess =qi.Session()
sess.connect(IP)
stand_speed=0.8
postureproxy = sess.service('ALRobotPosture')
postureproxy.goToPosture("StandInit", stand_speed)
import time


a = 0.4

image_name='ggg.jpg'
image = cv2.imread('IMM.jpg')
MotionProxy = sess.service('ALMotion')
MotionProxy.setAngles(['HeadPitch'],[a],1)
time.sleep(1)
connector = CVConnector(IP, camera_id=0)
ball = connector._get_ball(save_image=True, save_dir='PHOTO0.jpg')
x,y,w,h = ball




f = 600
L = 42
d = L / np.pi


c_x =    x + (w//2)  - 320
c_y = - (y + (h//2)) + 240

print (c_x, c_y)


D = f/w*d

Q, q = position_calc(0, a, c_x, c_y)

L = Q - q

A = L - q / q[2] * L[2]
A1 = A * np.sqrt(D**2 - L[2]**2) /np.linalg.norm(A)
print (A)
print (A1)

X = A[0]
Z = L[2]
Y = np.sqrt(D**2 - Z**2 - X**2)
print (X, Y, Z)
print (D - np.sqrt(Y**2 + Z**2 + X**2))


