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

marker_color_min = (0, 240, 135)
marker_color_max = (20, 255, 205)

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
    max_times = 3
    
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
