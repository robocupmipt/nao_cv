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
#from enum import Enum

marker_color_min = (0, 240, 135)
marker_color_max = (20, 255, 205)

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
    for line in lines:
        if (line.xmax<X_checkpoint and line.xmax>left_xmax):
            left_line,left_xmax=line,  line.xmax
        elif (line.xmin>X_checkpoint and line.xmin<right_xmin):
            right_line,right_xmin = line, line.xmin
        else:
            center_y=(line.y_min + line.y_max)/2
            if center_y<center_ymin:
                horisontal_line,center_ymin=line,center_y
	return [left_line,horisontal_line,right_line]

def getChessboardCentre(image, pattern_size):#If we use chessboard
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
def shouldTurn(image, allowed_gamma=100, X_checkpoint=X_DIM//2,Y_checkpoint=Y_DIM//2):
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

def GetNewAngle(current_angle,turn_command, delta):
    '''
    turn_command - result of ShouldTurn function
    '''
    return current_angle + turn_command * delta





def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def find_marker (img, color):
    hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, color[0], color[1])
    
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    
    marker_coord = [-5, -5]
    
    print (dArea)
    
    if dArea > 20:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
        marker_coord[0] = x
        marker_coord[1] = y
    
        #cv2.circle (img, (x, y), 10, (0,0,255), -1)
        #cv2.imshow ("detected", img)
    
    return marker_coord

def find_marker_chessboard (img):
    print ("Not implemented yet")

#class Turns(Enum):
#    RIGHT   = 1
#    LEFT    = 2
#    NO_TURN = 3

#MAX_ACCEPTBLE_EXCENTRICITY = 40

#def turn_needed (marker_coords):
#    if (marker_coords [0] < WIND_X / 2 - MAX_ACCEPTABLE_EXCENTRICITY):
#	return Turns.LEFT
	    
#    elif (marker_coords [0] < WIND_X / 2 + MAX_ACCEPTABLE_EXCENTRICITY):
#	return Turns.RIGHT

#    return Turns.NOTURN

#SHIFT_FROM_BOTTOM = 40

#def walk_back (marker_coords):
#    if (marker_coords [1] >= WIND_X - SHIFT_FROM_BOTTOM):
#	return True

#    return False

def main(robotIP):
    # Init proxies.
    try:
        motionProxy = ALProxy("ALMotion", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture"
        print "Error was: ", e

    # Set NAO in Stiffness On
    StiffnessOn(motionProxy)

    # Send NAO to Pose Init
    postureProxy.goToPosture("StandInit", 0.5)

    #####################
    ## Enable arms control by Walk algorithm
    #####################
    motionProxy.setWalkArmsEnabled(True, True)
    #~ motionProxy.setWalkArmsEnabled(False, False)

    #####################
    ## FOOT CONTACT PROTECTION
    #####################
    #~ motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", False]])
    motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True]])

    #TARGET VELOCITY
    #X = -0.5  #backward
    #Y = 0.0
    #Theta = 0.0
    Frequency =0.0 # low speed
    #motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

    #time.sleep(4.0)
    
    ###########################################

    #чтобы работал поворот
    tts = ALProxy("MovementGraph", "192.168.1.67", 9559)

    session = qi.Session()
    try:
        session.connect("tcp://" + "127.0.0.1" + ":" + "9559")

    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    
	#main(session)

    video_service = session.service("ALVideoDevice")

    #resolution = vision_definitions.kQQVGA
    #colorSpace = vision_definitions.kYUVColorSpace
    
    resolution = 2
    colorSpace = 11
    
    fps = 20
    
    nameId = video_service.subscribe("python_GVM", resolution, colorSpace, fps)
    
    #max_times = 20
    max_times = 5
    
    walk      = False
    walk_back = False

    print ("starting")

    for times in range (0, max_times):
	print (str (times) + " turn")	

	time_start = time.time ()
	img_pack = video_service.getImageRemote(nameId)
	print ("getImageRemote time: " + str (time.time () - time_start))

	time.sleep(0.05)
	    
	img_ = Image.frombytes("RGB", (img_pack [0], img_pack [1]), bytes(img_pack [6]))
	
	img = np.array (img_)	
	img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)

	marker_coords = find_marker (img, (marker_color_min, marker_color_max))	

	cv2.imwrite ("img" + str (times) + ".jpg", img)
	
	if (marker_coords [0] >= 0):
	    print ("marker detected")
	    walk = True

	    if (marker_coords [1] >= 440):
		walk_back = True

	    if (marker_coords [0] < 280):
		tts.Move (0, 0, 20)
	    
	    elif (marker_coords [0] > 360):
		tts.Move (0, 0, -20)

	else:
	    print ("no marker")
	    walk = False

	if (walk == True):
	    if (walk_back == False):
		print ("walking straight")

		X = 0.8
		Y = 0.0
		Theta = 0.0
		Frequency =1.0 # max speed
		motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)
	    
#	    else:
	if (walk_back == True):
		print ("walking back")

		X = -0.5  #backward
		Y = 0.0
		Theta = 0.0
		Frequency =0.0 # low speed
		motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)

	time.sleep (2.0)
	
    X = 0.0
    Y = 0.0
    Theta = 0.0
    motionProxy.setWalkTargetVelocity(X, Y, Theta, Frequency)
	
    time.sleep (1.0)

    print ("finished")

    postureProxy.goToPosture("StandInit", 0.5)

    video_service.unsubscribe(nameId)

if __name__ == "__main__":
    robotIp = "127.0.0.1"

    if len(sys.argv) <= 1:
        print "Usage python motion_walk.py robotIP (optional default: 127.0.0.1)"
    else:
        robotIp = sys.argv[1]

    main(robotIp)