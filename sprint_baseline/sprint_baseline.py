#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import qi
import argparse
import sys
import time
import cv2
import numpy as np
from PIL import Image
import os
#os.chdir('/home/robocup/nao')
TURNBACK_FLAG=False
STOP_FLAG = False
PREVIOUS_TURN=0
#from enum import Enum
#img image rename
marker_color_min = (0, 220, 90)#Min color of red box#(60,105)
marker_color_max = (20, 250, 135)#Max color of red box
pattern_size = (3,3)#Pattern size of chessboard

from line_detect import getLines,Line,X_DIM,Y_DIM
def processLines(raw_lines, X_checkpoint=X_DIM/2,Y_checkpoint=Y_DIM/2):
    left_xmax,right_xmin,center_ymin=0,9999,9999
    left_line,right_line,horisontal_line=None,None,None
    smallest_k = min([line.k for line in raw_lines if line.k<0]+[9999999])
    largest_k = max([line.k for line in raw_lines if line.k>0]+[-9999999])
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
    def canBeHorisontal(line):
        return not canBeLeft(line) and  not canBeRight(line)
    for line in raw_lines:
        if (canBeLeft(line) and line.xmax>left_xmax):
            left_line,left_xmax=line,  line.xmax
        elif (canBeRight(line) and line.xmin<right_xmin):
            right_line,right_xmin = line, line.xmin
        elif canBeHorisontal(line):
            center_y=(line.ymin + line.ymax)/2
            if center_y<center_ymin:
                horisontal_line,center_ymin=line,center_y
    return [left_line,horisontal_line,right_line]
def log(image, lines,thickness=5):
    t = str(time.time())
    cv2.imwrite('prev_'+t+'.jpg',image)
    colors=[(255,0,0),(0,255,0),(0,0,255)]
    for (line,color) in zip(lines,colors):
        if line is not None:
            cv2.line(image,
            (line.xmin,line.y(line.xmin)),
            (line.xmax,line.y(line.xmax)),color,thickness)
    cv2.imwrite(t+'.jpg',image)
def getChessboardCentre(image, pattern_size=pattern_size):#If we use chessboard
    '''
    image is np.array, pattern_size istuple
    '''
    ans=None
    for i in range(3, pattern_size[0]+1):
        for j in range(3,pattern_size[1]+1):
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
        print('COORDINATES FOUND')
        print(str(marker_coord))
    else:
        print('COORDINATES NOT FOUND')
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
               

def ObjectCondition(center_x, allowed_gamma=100):
    global PREVIOUS_TURN
    print('Center_X is'+str(center_x))
    if center_x<0:
        return PREVIOUS_TURN
    if center_x < X_DIM/2 - allowed_gamma:
        print('To turn left')
        PREVIOUS_TURN=-1
        return -1
    elif center_x>X_DIM/2 + allowed_gamma:
        print('To turn right')
        PREVIOUS_TURN=1
        return 1
    else:
        PREVIOUS_TURN=0
        print('Not to turn')
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
def ShouldTurn(image, allowed_gamma=100, X_checkpoint=X_DIM//2,Y_checkpoint=Y_DIM//2,
               log_lines= True, use_lines=False):
    global TURNBACK_FLAG
    if use_lines:
        raw_lines = getLines(image)#list of objects class Line
        left_line,right_line,horisontal_line = processLines(raw_lines, X_checkpoint,Y_checkpoint)
        if log_lines:
            log(image,[left_line,right_line,horisontal_line])
        center = getChessboardCentre(image)#maybe other function and other object instead of chessboard
        back_condition = BackCondition(horisontal_line,center)
        line_condition = LineCondition(left_line,right_line)
        if back_condition:
            TURNBACK_FLAG=True
    else:
        line_condition=0
    center = getObjectCentre(image,'redbox')
    object_condition = ObjectCondition(center[0],allowed_gamma)
    if abs(object_condition)>0:
         return object_condition
    if abs(line_condition)>0:
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


def subscribe(video_service, camera_id=0,
              resolution=2, colorSpace=11,fps=5):
    ###Subscribes videoservice to camera
    ###I'm not sure if last parameter really stands for fps - it needs to be checked
    return video_service.subscribeCamera('python_client',camera_id,resolution, colorSpace, fps)

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
time_lag,stand_speed=2,0.6
#robotIP='192.168.1.2'

#from tqdm import tqdm

robotIP='192.168.8.115'
robotIP='192.168.1.12'
disable_state=True#DISABLE BEHAVIOR IN SCRIPT
make_photos=False#MAKE PHOTOS IN SCRIPT
n_chunks=10#NUMBER OF CHUNKS AT THE END OF EACH STOPS TO MAKE PHOTO
extra_lag=1.5
expected_speed=0.06
def mainBlind(robotIP, disable_state=disable_state,make_photos=make_photos, n_chunks=n_chunks,
              speed=expected_speed,extra_lag=extra_lag):
z
    print('going back')
    time.sleep(2)
    time_chunk = time_chunk/2
    postureProxy.goToPosture("StandInit", 0.6)
    customProxy.StartMove()
    for i in (range(n_chunks)):
        customProxy.GoBack(walk_dist)#walk walk_dist meters forward
        time.sleep(time_chunk*back_ntimes_slower)
        if make_photos:
            motionProxy.setAngles(["HeadPitch","HeadYaw"],[0,0],0.3)
            img_pack = video_service.getImageRemote(video_client)
            assert img_pack is not None
            img = np.array(Image.frombytes("RGB",(img_pack[0],img_pack[1]),bytes(img_pack[6])))
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('PHOTO_'+str(i)+'.jpg',image)
            turn_flag = ShouldTurn(image, allowed_gamma=100)
            if turn_flag:
                print('Turning')
                motionProxy.moveTo(0,0,rotate_theta*turn_flag*np.pi/180)
        #Turn on angle
        customProxy.StopMove()
        time.sleep(extra_lag)
    postureProxy.goToPosture("StandInit", 0.6) 
    
if __name__ == "__main__":
    robotIp  = '127.0.0.1'#sys.argv[1]
    #mainBlind(robotIp)
    mainBlind(robotIp)
#OLDER FUNCTION
def main(robotIP, time_lag = 2, stand_speed= 0.5):
#    # Init proxies.
#    #motionProxy, customProxy, postureProxy, video_service = make_proxies(robotIP,9559,False)
#    proxy_list=[]
#    proxy_names,port=['ALMotion','MovementGraph','ALRobotPosture',
#                   'ALVideoDevice'],9559
#    sess = qi.Session()
#    try:
#        sess.connect("tcp://"+robotIP+":"+str(port))
#        for proxy_name in proxy_names:
#            proxy_list.append(sess.service(proxy_name))
#    except:
#        print('Can not connect at ip '+robotIP+' and port '+str(port))
#        raise Exception()
#    motionProxy, customProxy, postureProxy, video_service = proxy_list
#    StiffnessOn(motionProxy)
#    motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True],
#                                ['MaxStepX',0.06],['StepHeight',0.027],
#                                 ['TorsoWy',0.01]])
#    motionProxy.setMoveArmsEnabled(True,True)
#    postureProxy.goToPosture("StandInit", stand_speed)
#
#    
#    video_client = subscribe(video_service, 0)
#    prev_time = 0
#    n_times = 5    
#    print ("starting")
#    for i in range((n_times)):
#        this_time = time.time()
#        if this_time>prev_time+time_lag:
#            img = get_photo(video_service, video_client, photo_ind=i)
#            prev_time = time.time()
#        turn_command = ShouldTurn(img, allowed_gamma = 100,log_lines=True)
#        GetNewAngle(customProxy, turn_command)
#        if STOP_FLAG:
#            motionProxy.setWalkTargetVelocity(0,0,0,0)
#        elif TURNBACK_FLAG:
#            motionProxy.setWalkTargetVelocity(-0.5,0,0,1)
#        else:
#            motionProxy.setWalkTargetVelocity(0.8,0,0,1)
#    motionProxy.unsubscribe(postureProxy,video_service, video_client)
##def TurnCriteria(photo):
#    center = getObjectCentre(image,'redbox')