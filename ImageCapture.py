#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:35:47 2019

@author: robocup
"""
import qi
from PIL import Image
IP='192.168.1.13'
PORT=9559
resolution = 2
CameraIndex=0   # VGA
colorSpace = 11 
ses = qi.Session()
ses.connect(IP)
YAW,PITCH=0,0
video= ses.service('ALVideoDevice')
motionProxy  = ses.service('ALMotion')
postureProxy = ses.service('ALRobotPosture')
videoClient = video.subscribeCamera(
        "python_client", CameraIndex, resolution, colorSpace, 5)
image_name='6.jpg'
def get_photo(
              image_name=None,pitch=0,yaw=0,speed=0.1,
              save_image=True
              ):
    '''
    pitch - vertical angle(in RAD, from -0.67 to 0.51 )
    yaw - horisontal angle(in RAD, from -2.08 to 2.08)
    speed - speed of placing angles to this place
    cameraIndex  -index of camera ( 0 is top camera, 1 is bottom camera)
    save_image : whether we save image or not
    image_name - name used for saving(if not specified, default name is generated)
    '''
    if not image_name:
        image_name='IMAGE_'+str(yaw)+'_'+str(pitch)+'.jpg'
#    motionProxy.setAngles(["HeadPitch","HeadYaw"],[pitch,yaw],speed)#

    naoImage = video.getImageRemote(videoClient)

        
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    im = Image.frombytes("RGB", (imageWidth, imageHeight), str(array))
    if save_image:
        im.save(image_name)
    return im
import sys
arguments=sys.argv[1:]
default_args=[None,0,0,0.1,True]
for i in range(len(arguments)):
    default_args[i]=arguments[i]
ff=get_photo(default_args[0],default_args[1],default_args[2],default_args[3],default_args[4])
video.unsubscribe(videoClient)
