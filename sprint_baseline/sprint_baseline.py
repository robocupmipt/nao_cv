# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:05:12 2019

@author: user
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:56:51 2019
@author: asus
"""

# -*- encoding: UTF-8 -*- 

'''Walk: Small example to make Nao walk'''
import sys
import motion
import time
from naoqi import ALProxy
import qi
import argparse
import sys
import time
import vision_definitions
import cv2
import numpy as np
from PIL import Image
TURNBACK_FLAG=False
STOP_FLAG = False
#from enum import Enum

marker_color_min = (0, 240, 135)#Min color of red box
marker_color_max = (20, 255, 205)#Max color of red box
pattern_size = (8,6)#Pattern size of chessboard
X_DIM = 640
Y_DIM = 480
class Line():
    def __init__(self,k, b, x_min,x_max):
        self.k,self.b=k,b
        self.x_min,self.x_max = x_min,x_max
        y_extrs = [k*x_min+b,k*x_max+b]
        self.y_min,self.y_max = min(y_extrs),max(y_extrs)
def getLines(image):
    pass
def processLines(lines, X_checkpoint=X_DIM/2,Y_checkpoint=Y_DIM/2):
    left_xmax,right_xmin,center_ymin=0,9999,9999
    left_line,right_line,horisontal_line=None,None,None
    smallest_k = min([line.k for line in lines if line.k<0])
    largest_k = max([line.k for line in lines if line.k>0])
    n=2
    def canBeLeft(line):
        cond1 = (
        line.xmax - line.xmin<(X_DIM/6) and (line.xmax+line.xmin)/2<X_DIM/2 )
        cond2 = (line.k>largest_k/n)
        return cond1 or cond2
    def canBeRight(line):
        cond1 = (
        line.xmax - line.xmin<(X_DIM/6) and (line.xmax+line.xmin)/2<X_DIM/2 )
        cond2 = (line.k<smallest_k/n)
        return cond1 or cond2
    
    for line in lines:
        if (canBeLeft(line) and line.xmax>left_xmax):
            left_line,left_xmax=line,  line.xmax
        elif (canBeRight(line) and line.xmin<right_xmin):
            right_line,right_xmin = line, line.xmin
        else:
            center_y=(line.y_min + line.y_max)/2
            if center_y<center_ymin:
                horisontal_line,center_ymin=line,center_y
    return [left_line,horisontal_line,right_line]

def getChessboardCentre(image, pattern_size=pattern_size):#If we use chessboard
    '''
    image is np.array, pattern_size istuple
    '''
    ans=None
    for i in range(3, pattern_size.shape[0]+1):
        for j in range(3,pattern_size.shape[1]+1):
            found, corners = cv2.findChessboardCorners(image,(i,j),None)
            if found:
                ans = corners.sum(0)/len(corners)
    return ans
def getRedboxCentre (image, color = (
        marker_color_min, marker_color_max),
    min_d_area=20):
    hsv = cv2.cvtColor (image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, color[0], color[1])
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']  
    marker_coord = [-5, -5]    
    print (dArea)
    if dArea > min_d_area:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
        marker_coord[0] = x
        marker_coord[1] = y    
    return marker_coord

def getObjectCentre(image,mode='chessboard'):
    '''
    Mode must be either 'chessboard' or 'redbox'
    '''
    if mode.lower()=='chessboard':
        return getChessboardCentre(image)
    elif mode.lower()=='redbox':
        return getRedboxCentre(image)
    else:
        raise Exception('Unknown mode '+str(mode))
               

def ObjectCondition(center_y, allowed_gamma=100):
    if center_y < Y_DIM/2 - allowed_gamma:
        return -1
    elif center_y>Y_DIM/2 + allowed_gamma:
        return 1
    else:
        return 0
def LineCondition(left_line, right_line):
     if left_line is None and right_line is not None:
         return 1
     if right_line is None and left_line is not None:
        return -1
     return 0         
     
def BackCondition(line, center):
    if line is None or line.y_min>center.y:
        return 1
    return 0
def ShouldTurn(image, allowed_gamma=100, X_checkpoint=X_DIM//2,Y_checkpoint=Y_DIM//2):
    global TURNBACK_FLAG
    raw_lines = getLines(image)#list of objects class Line
    left_line,right_line,horisontal_line = processLines(raw_lines, X_checkpoint,Y_checkpoint)
    center = getChessboardCentre(image)#maybe other function and other object instead of chessboard
    back_condition = BackCondition(horisontal_line,center)
    line_condition = LineCondition(left_line,right_line)
    object_condition = ObjectCondition(center[1],allowed_gamma)
    if back_condition:
        TURNBACK_FLAG=True
    if abs(object_condition)>1:
         return object_condition
    if abs(line_condition)>1:
         return line_condition
    return 0

def GetNewAngle(customProxy,turn_command, delta=20):
    '''
    turn_command is a result of ShouldTurn function
    delta is an angle we should rotate on
    '''
    if abs(turn_command)>1e-2:
        customProxy.Move(0,0,turn_command*delta)

def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)


def make_proxies(robotIP,port=9559, session=False):
    '''
    if session is False, ALProxys are created. 
    Otherwise, connections through qi.Session are created.

    '''
    proxy_names = ['ALMotion','MovementGraph','ALRobotPosture',
                   'ALVideoDevice']
    proxy_list=[]
    if not session:
        for proxy_name in proxy_names:
            try:
                proxy_list.append(
                ALProxy(proxy_name, robotIP,port))
            except Exception as e:
                print('Could not create proxy to '+proxy_name)
                print('Error was: '+str(e))
                raise Exception()
    elif session:
        sess = qi.Session()
        try:
            sess.connect("tcp://"+robotIP+":"+str(port))
            for proxy_name in proxy_names:
                proxy_list.append(sess.service(proxy_name))
        except:
            print('Can not connect at ip '+robotIP+' and port '+str(port))
            raise Exception()
    return proxy_list
def subscribe(video_service, camera_id=0,
              resolution=2, colorSpace=11,fps=5):
    ###Subscribes videoservice to camera
    ###I'm not sure if last parameter really stands for fps - it needs to be checked
    return video_service.subscribeCamera(camera_id,resolution, colorSpace, fps)

def get_photo(video_service,video_client, photo_ind = 0, time_est=True):
    '''
    Note: video_service must be first and video_client second. Not vice versa
    '''
    t=time.time()
    img_pack = video_service.getImageRemote(video_client)
    assert img_pack is not None
    img = np.array(Image.frombytes("RGB",(img_pack[0],img_pack[1]),bytes(img_pack[6])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if time_est:
        print('Photo received for '+str(time.time()-t))
        cv2.imwrite ("img" + str (photo_ind) + ".jpg", img)
    return img
def finish(postureProxy, video_service,video_client):
    time.sleep(1.0)
    print('finished')
    video_service.unsubscribe(video_client)

def main(robotIP, time_lag = 2, stand_speed= 0.5):
    # Init proxies.
    motionProxy, customProxy, postureProxy, video_service = make_proxies(robotIP,9559,False)
    
    StiffnessOn(motionProxy)
    motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])
    postureProxy.goToPosture("StandInit", stand_speed)
    
    video_client = subscribe(video_service, 0)
    prev_time = 0
    n_times = 5    
    print ("starting")
    for i in range(len(n_times)):
        this_time = time.time()
        if this_time>prev_time+time_lag:
            img = get_photo(video_service, video_client, photo_ind=i)
            prev_time = time.time()
        turn_command = ShouldTurn(img, allowed_gamma = 100)
        GetNewAngle(customProxy, turn_command)
        if STOP_FLAG:
            motionProxy.setWalkTargetVelocity(0,0,0,0)
        elif TURNBACK_FLAG:
            motionProxy.setWalkTargetVelocity(-0.5,0,0,1)
        else:
            motionProxy.setWalkTargetVelocity(0.8,0,0,1)
    motionProxy.unsubscribe(postureProxy,video_service, video_client)

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print ("Usage python motion_walk.py robotIP (optional default: 127.0.0.1)")
    else:
        robotIp = sys.argv[1]

    main(robotIp)
