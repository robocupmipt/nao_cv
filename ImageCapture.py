#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:35:47 2019

@author: robocup
"""
import qi
from PIL import Image
IP='192.168.1.14'
PORT=9559
resolution = 2
CameraIndex=1   # VGA
colorSpace = 11 
ses = qi.Session()
ses.connect(IP)
YAW,PITCH=0,0
video= ses.service('ALVideoDevice')
motionProxy  = ses.service('ALMotion')
postureProxy = ses.service('ALRobotPosture')
videoClient = video.subscribeCamera(
        "python_client", CameraIndex, resolution, colorSpace, 5)
def get_photo(pitch=0,yaw=0,speed=0.1,
              cameraIndex=1,
              image_name=None,
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
    motionProxy.setAngles(["HeadPitch","HeadYaw"],[pitch,yaw],speed)#

    naoImage = video.getImageRemote(videoClient)

        
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    im = Image.frombytes("RGB", (imageWidth, imageHeight), str(array))
    if save_image:
        im.save(image_name)
    return im
ff=get_photo()
video.unsubscribe(videoClient)
