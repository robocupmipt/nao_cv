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
    if not image_name:
        image_name='IMAGE_'+str(yaw)+'_'+str(pitch)+'.jpg'
    if set_angles:
        motionProxy.setAngles(["HeadPitch","HeadYaw"],[pitch,yaw],speed)
    for client, i in zip([videoClient0,videoClient1],['0','1,']):
        naoImage = video.getImageRemote(client)
        im = Image.frombytes("RGB", (naoImage[0],naoImage[1]), bytes(naoImage[6]))
        if save_image:
            im.save('image'+str(image_index)+'cam'+str(i)+'_'+str(yaw)+'_'+str(pitch)+'.jpg'
ss=0
while True:
   print('Нажмите 0 для выхода, иное число для съема ')
   get_photo(ss)
   ss+=1

print('Все')
 