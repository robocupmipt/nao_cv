

from PIL import Image
from tqdm import tqdm
import qi
import numpy as np
import cv2 as cv
import os
os.chdir('/home/robocup/many_chessboards')#
IP='192.168.1.15'
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

GENERATE_IMAGES=True
if GENERATE_IMAGES:
        
    IP='192.168.1.5'
    PORT=9559
    resolution = 2    # VGA
    colorSpace = 11
    X,Y=50,50
    for YAW in tqdm(np.arange(-0.5,0.5,0.05)):
       for PITCH in np.arange(-0.6,0.5,0.03):
           
           motionProxy.setAngles("HeadYaw",YAW,0.7)#
           motionProxy.setAngles("HeadPitch",PITCH,0.7)
           arr = video.getImageRemote(videoClient)
           naoImage = Image.frombytes('RGB',(arr[0],arr[1]),
           arr[6])
           naoImage = np.array(naoImage)
           (found,corners)=cv.findChessboardCorners(naoImage,(9,6),
            flags =cv.CALIB_CB_ADAPTIVE_THRESH|cv.CALIB_CB_FILTER_QUADS,
            stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30, 0.001))
           if found and len(corners)==54:
               cv.imsave('IMAGE_'+str(X)+'_'+str(Y)+'_'+str(YAW)+'_'+str(PITCH)+'.jpg',naoImage)
