import os
import cv2
import numpy as np

os.chdir(r'C:\Users\Vovan\\Downloads\floor')


sqrt = np.sqrt
sin  = np.sin
cos  = np.cos
tan  = np.tan
pi   = np.pi
arr  = np.array
asin = np.arcsin
acos = np.arccos
atan = np.arctan
norm = np.linalg.norm


def cast(a):    
    '''
    Gets output of cv2.findChessboardCorners
    Returns an array of corners to necessary shape
    '''
    b = np.zeros([len(a), 2])
    for i in range(len(b)):
        b[i] = a[i][0]
    return b


def corners(b):
    '''
    Gets custed output of cv2.findChessboardCorners
    Returns edge corners of the desk
    '''
    b = sorted(b, key = lambda x: x[1])                    #sorting points by Y axis
    
    b1 = np.around(sorted(b[:7],  key = lambda x: x[0]))   #extraction of top cornters 
    b2 = np.around(sorted(b[-7:], key = lambda x: x[0]))   #extraction of bottom corners
    
    #three corners are enough
    return arr([b2[0], b1[0], b2[-1]])      



def pix_trans(b):
    '''
    Gets coordinates in the image in left top corner reference system
    Returns coordinates of projection of light rays on photomatrix in central reference system
    '''
    window_size = [640, 480, 0]

    b_const = 1/2 * (window_size * arr([-1, 1, 0])) * np.ones([3, 3])

    b_coef  = arr([1, -1, 0]) * np.ones([3, 3]) 

    #invert (*-1) because lens invert image
    return - (b_coef * b + b_const)    


def pixels2rays(points, f):
    '''
    Gets array of points  [n, 3] like [x, y, 0]
    Cast points on the photomatrix in central reference system to components of light ray vector 
    knowing optical vector f
    Return array of light [n, 3]
    '''
    x0 = arr([-f[1], f[0], 0])    #x axis on the image in central reference system orthogonal to f
    x0 = x0/norm(x0)
    
    y0 = arr([-f[0]*f[2], -f[1]*f[2], f[0]**2+f[1]**2])
    y0 = y0/norm(y0)                 #y axis on the image in central reference system orthogonal  to f
    
    
    C = np.zeros([3,3])   #transformation matrix
    C[0] = x0
    C[1] = y0
    C = np.transpose(C)
    
    rays = np.zeros(points.shape)
    for i in range(len(rays)):
        rays[i] = C.dot(points[i]) + f
        
    return rays

def real_coordinates(pixels, foc, Len):
    '''
    Gets array of points in central reference system, optical vector, position of the lens
    Returns coordinates of the points on the floor in robot's reference system
    '''
    
    ray = pixels2rays(pixels, foc)      #cast pixels to light rays
    
    real_points = np.zeros(ray.shape)   #find coordinates of points on the floor
    for i in range(len(real_points)):
        real_points[i] = Len - Len[2]/ray[i][2] * ray[i]
        
    return real_points

def calibration1(size, am): 
    
    '''
    Gets size of the desk and point on photomatrix
    Returns 3d coordinates of the lens in the space - L, and componets of optical axis of the lens - f
    norm of f approximately shows distance between lens and photomatrix
    '''
    
    x1 = am[0][0]   #necessary variables
    x2 = am[1][0]
    x3 = am[2][0]

    y1 = am[0][1]
    y2 = am[1][1]
    y3 = am[2][1]

    X1 = x1 - x2
    X2 = x2 - x3
    X3 = x3 - x1

    Y1 = y1 - y2
    Y2 = y2 - y3
    Y3 = y3 - y1

    Z1 = y1*x2 - y2*x1
    Z2 = y2*x3 - y3*x2
    Z3 = y3*x1 - y1*x3
    
    dx1 = X1/Z1 - X2/Z2
    dx2 = X2/Z2 - X3/Z3
    dx3 = X3/Z3 - X1/Z1

    dy1 = Y1/Z1 - Y2/Z2
    dy2 = Y2/Z2 - Y3/Z3
    dy3 = Y3/Z3 - Y1/Z1

    
    k3 = - Y1/Z1 * dx2                 #coefficients of third degree polynomial ki where i - power of argument
    k2 =   Y1/Z1 * dx1 + dy1 * dx3
    k1 = - Y3/Z3 * dx2 + dy2 * dx3 
    k0 =   Y3/Z3 * dx1

   
    cb = np.roots([k3, k2, k1, k0])             #find polynomial zeros. cb - tan of turn angle

    nans = []
    
    for i in range(len(cb)):                    #delete imaginary roots 
        if (np.imag(cb[i]) != 0):
            nans += [i]
            
    cb = np.real(np.delete(cb, nans))
    
    
    cc = (cb * dy1 + dy2) / (cb * dx2 - dx1)     #find cc - sin of incline angle
    
    
    nans = []                                    #delete solution > 1 by abs value. sin can't be > 1
    for i in range(len(cc)):                     #delete solution > 0. angle should be < 0. camera pointing down
        if ((abs(cc[i]) > 1)  or (cc[i] > 0)):
            nans += [i]
            
    cc = np.delete(cc, nans)      #delete cc unsuitable roots and cb both
    cb = np.delete(cb, nans)    
    
    
    
    ca = np.zeros(len(cb))       #find ca - norm of optical vector f
    for i in range(len(ca)):
        ca[i] = - (sqrt(1 -(cc[i])**2) * Z2 / (cc[i] * X2 + (cb[i]-1) / (cb[i]+1) * Y2))
    
   
    nans = []                    #delete solutions < 0. norm can't be < 0
    for i in range(len(ca)):
        if (ca[i] < 0):
            nans += [i]
    
    ca = np.delete(ca, nans)     #delete unsiutable solutions for all parameters
    cb = np.delete(cb, nans)
    cc = np.delete(cc, nans)

    
    prams = np.transpose(arr([ca, atan(cb), asin(cc)]))    #collecte all parameters to an array
    p = np.real(prams)                                     #delete imaginary part(=0) to not spoil output   
    
    
    f = np.zeros(p.shape)      #find components of optical vector for all posible solutions
    
    for i in range(len(p)):
        f[i] = -p[i][0]*arr([-cos(p[i][2])*sin(p[i][1]), cos(p[i][1])*cos(p[i][2]), sin(p[i][2])])
    
    
    q = np.zeros([len(f), 3, 3])         #find components of the vector of light ray
    for i in range(len(f)):
        q[i] = pixels2rays(am, f[i])     #cast coordinates of points on the photomatrix to components of ray vectors

        
    L = np.zeros(f.shape)        #find coordinates of the lens
    for i in range(len(L)):
        L[i][2] = size / (q[i][0][0]/q[i][0][2] - q[i][2][0]/q[i][2][2])
        L[i][1] = L[i][2] * q[i][0][1] / q[i][0][2]
        L[i][0] = L[i][2] * q[i][0][0] / q[i][0][2]
    
    return f, L




image0 = cv2.imread('IMAGE_0_0.0_35.jpg')      #reading imagines
image1 = cv2.imread('IMAGE_0_0.1_35.jpg')
image2 = cv2.imread('IMAGE_0_0.2_35.jpg')
image3 = cv2.imread('IMAGE_0_0.3_35.jpg')

start_cors = [[], [], []]     #collecting all corners


_,start_cors[0] = cv2.findChessboardCorners(image0,(7,7))  #find corners of all squares on the image
_,start_cors[1] = cv2.findChessboardCorners(image1,(7,7))
_,start_cors[2] = cv2.findChessboardCorners(image2,(7,7))


cors = np.zeros([len(start_cors), 3, 3])           #array for desk corners. three corners of three coordinates

for i in range(len(start_cors)):                   #extraction of desk corners
    cors[i][:,:-1] = corners(cast(start_cors[i]))
    
cors = arr(cors)
    
pixels = pix_trans(cors)                           #cast image coordinates to normal reference system 

    
points = pix_trans(cors)

size = 51/8*7

for j in range(len(pixels)):
    pix = points[j] 
    f, L = calibration1(size, pix)

    for i in range(len(f)):
        print ('|f|   = ', norm(f[i])) 
        print ('angle = ', 180/pi*(-acos(f[i][1]/norm(f[i]))+pi))
        print ('L     = ', L[i])
        print ('f     = ', f[i])
        print ()
print ()

