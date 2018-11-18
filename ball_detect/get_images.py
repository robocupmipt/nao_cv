# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import time
import math
# Python Image Library
from PIL import Image

from naoqi import ALProxy
IP='192.168.1.2'
PORT=9559
camProxy = ALProxy("ALVideoDevice", IP, PORT)
motionProxy  = ALProxy("ALMotion", IP, PORT)
postureProxy = ALProxy("ALRobotPosture", IP, PORT)
resolution = 2    # VGA
colorSpace = 11   # RGB

motionProxy.wakeUp()
postureProxy.goToPosture("StandInit", 0.2)
effectorName = "Head"
# Active Head tracking
isEnabled    = True
motionProxy.wbEnableEffectorControl(effectorName, isEnabled)
videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

motionProxy.setAngles("HeadYaw",-1,0.1)

motionProxy.setAngles("HeadPitch",-0.2,1)

naoImage = camProxy.getImageRemote(videoClient)
camProxy.unsubscribe(videoClient)
imageWidth = naoImage[0]
imageHeight = naoImage[1]
array = naoImage[6]
im = Image.frombytes("RGB", (imageWidth, imageHeight), array)
im.save('BALL1.jpg','JPG')



    # Wake up robot
 

    # Deactivate Head tracking

