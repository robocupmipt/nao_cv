import qi
from PIL import Image
import cv2
import os
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
customMotionProxy = ses.service('MovementGraph')
postureProxy = ses.service('ALRobotPosture')
videoClient0 = video.subscribeCamera(
        "python_client", 0, resolution, colorSpace, 5)
videoClient1 = video.subscribeCamera(
        "python_client", 1, resolution, colorSpace, 5)
image_name='6.jpg'
def get_photo(
              image_index=0,pitch=0,yaw=0,speed=0.1,
              save_image=True,set_angles=False,video=video
              ):
    '''
    pitch - vertical angle(in RAD, from -0.67 to 0.51 )
    yaw - horisontal angle(in RAD, from -2.08 to 2.08)
    speed - speed of placing angles to this place
    cameraIndex  -index of camera ( 0 is top camera, 1 is bottom camera)
    save_image : whether we save image or not
    image_name - name used for saving(if not specified, default name is generated)
    '''
    if set_angles:
        motionProxy.setAngles(["HeadPitch","HeadYaw"],[pitch,yaw],speed)
    for client, i in zip([videoClient0,videoClient1],['0','1,']):
        naoImage = video.getImageRemote(client)
        im = Image.frombytes("RGB", (naoImage[0],naoImage[1]), bytes(naoImage[6]))
        if save_image:
            im.save('image'+str(image_index)+'cam'+str(i)+'_'+str(yaw)+'_'+str(pitch)+'.jpg')
ss=0
while True:
   print('Нажмите 0 для выхода, иное число для съема ')
   get_photo(ss)
   ss+=1
 
from line_detection import getLines
from sprint_baseline import processLines, getObjectCentre
def log(image, lines,other_lines, center, image_name, thickness=1, circle_range=10):
    colors=[(255,0,0),(0,255,0),(0,0,255)]
    if center[0]>0:
         cv2.circle(image, center, circle_range, colors[1])
    for line in other_lines:
        cv2.line(image,
            (line.xmin,line.y(line.xmin)),
            (line.xmax,line.y(line.xmax)),(0,0,0),thickness)
    for (line,color) in zip(lines,colors):
        if line is not None:
            cv2.line(image,
            (line.xmin,line.y(line.xmin)),
            (line.xmax,line.y(line.xmax)),color,thickness)         
    cv2.imwrite(''+image_name+'EDITED.jpg',image)
condition0=('.jpg' in image_name and 'IMAGE' in image_name)#photos made in Moscow
condition1=('.jpg' in image_name and 'cam' in image_name)#photos made here
for image_name in os.listdir(os.getcwd()) :
    if condition0:
        image = cv2.imread(image_name)
        other_lines = getLines(image)
        lines = processLines(other_lines)
        center = getObjectCentre(image,mode='redbox')
        log(image,lines,other_lines,center, image_name)
 walk_dist=1#meters
n_times=3#times

print('Все, дальше тестируем ходьбу - введите что угодно и поставьте робота так чтобы он мог пройти '+str(walk_dist*n_times)+' метров вперед')
z=input()

def prepare(motionProxy, postureProxy, stand_speed=0.5):
      #Подготовка прокси к движению
      StiffnessOn(motionProxy) 
      motionProxy.setMotionConfig([["ENABLE_FOOT_CONTACT_PROTECTION", True], 
      ['MaxStepX',0.06],['StepHeight',0.027], ['TorsoWy',0.01]]) 
      motionProxy.setMoveArmsEnabled(true,true) 
      postureProxy.goToPosture("StandInit", stand_speed)
USE_CUSTOMPROXY=True#юзаем свое
if USE_CUSTOMPROXY:
    used_proxy=customMotionProxy
else:
    used_proxy=motionProxy

prepare(used_proxy, motionProxy)

#Начинаем тестирование
for i in range(2):
    for k in range(n_times):
        t=time.time()
        if i>0:
             used_proxy.()#walk walk_dist meters forward
             dir_='forward'
        else:
             used_proxy.()#walk walk_dist meters backward
             dir_='backward'
        print('Walking for '+str(walk_dist)+' m '+str(dir)+' took '+str(time.time()-t)+' sec')


