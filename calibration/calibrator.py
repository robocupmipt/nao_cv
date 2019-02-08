# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:58:26 2018

@author: user
"""
from tqdm import tqdm
import numpy as np
import cv2 as cv
import os
os.chdir('C:/Users/user/Downloads/many_chessboards(NEW)')
image_names=os.listdir(os.getcwd())

N_SAMPLES=[len(image_names)*3//4]
#img = cv2.imread(fname)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray.shape[::-1]
#gray.shape
camera_matrix = np.zeros((3, 3),'float32')

camera_matrix=[[ 1.46240075e+03,  0.00000000e+00,  3.24806002e+02],
 [ 0.00000000e+00,  1.00206827e+03, -6.17942773e+00],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

dist_coefs=[[ 1.22394222, -2.11858604, -0.21379144, -0.04868427,  3.25447246]]
for sample_num in N_SAMPLES:
    indexes = np.random.choice(len(image_names), sample_num, replace=False)
    image1_names = [image_names[i] for i in indexes]

    CHESSBOARD_SIZE=(9,6)
    CELL_SIZE = 2.65#CM
    
    objectpoints_total=[]
    imagepoints_total=[]
    import time
    t=time.time()
    for image_name in tqdm(image1_names):
       gray_image = cv.imread(image_name,0)
       object_points=[]
       try:
           (found,corners)=cv.findChessboardCorners(gray_image,CHESSBOARD_SIZE,
           flags =cv.CALIB_CB_ADAPTIVE_THRESH|cv.CALIB_CB_FILTER_QUADS)
           stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                             30, 0.001)#or 0.1
           image_points=[]
           point_indices=[]
           if found:
               point_indices = cv.cornerSubPix(gray_image, corners,
                        (11,11), (-1,-1), stop_criteria)
           if len(point_indices)==CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1]:
               for i in range(len(point_indices)):
                    image_points.append([point_indices[i][0][0],point_indices[i][0][1]])
                    object_points.append([i%CHESSBOARD_SIZE[0], i//CHESSBOARD_SIZE[1],0])
               object_points=np.array(object_points).astype('float32')
               image_points=np.array(image_points).astype('float32')
               objectpoints_total.append(object_points)
               imagepoints_total.append(image_points)
           else:
               os.remove(image_name)
       except:
            os.remove(image_name)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
    objectpoints_total, imagepoints_total,gray_image.shape[::-1],None,None)
    print('SAMPLE SIZE '+str(sample_num))
    print(time.time()-t)
    print('Directory')
    print(os.getcwd())
    print('camera_matrix')
    print(camera_matrix)
    print('rms')
    print(rms)
    print('dist_coefs')
    print(dist_coefs)
    import _pickle as cPickle
    cPickle.dump((rms, camera_matrix, dist_coefs, rvecs, tvecs),open('calib_data2.pkl','wb'))

'''
Диагональные коэфф-ты должны быть примерно равны друг другу
последние коэффициенты дисторсии должы быть маленькими
ВАЖНО, ЧТО НУЖНО БРАТЬ shape[::-1] !!!!!
MANYDIST = Ввыборка 7


Выборка 7 (верный подход к shape, консоль 24А)

Предварительно на 432
camera_matrix
[[1.50526410e+03 0.00000000e+00 3.68806157e+02]
 [0.00000000e+00 9.53438871e+02 9.64315180e+01]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
4.933648421214757
dist_coefs
[[ 1.61212328 -7.22800618 -0.08471607 -0.08278571 25.02718083]]

Выборки 1,4(консоль 27А)

Directory
C:\Users\user\Downloads\many_chessboards14
camera_matrix
[[1.90647358e+03 0.00000000e+00 3.19745443e+02]
 [0.00000000e+00 1.80349894e+03 1.56042337e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
8.40679505434177
dist_coefs
[[ 2.77371885e+00 -2.86595154e+01 -8.44277083e-02 -3.24702463e-02
   7.26861870e+02]]


Выборка 6 (консоль 28А)

camera_matrix
[[1.13837380e+04 0.00000000e+00 3.41587189e+02]
 [0.00000000e+00 3.27591274e+03 2.21675525e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.1047639922229395
dist_coefs
[[ 2.40885209e+01 -1.41047909e+03  9.71122182e-01 -2.28036264e-02
  -2.70913949e+03]]

Выборка 5(консоль 29А)
SAMPLE SIZE 869
200951.77968478203
Directory
C:\Users\user\Downloads\many_chessboards5
camera_matrix
[[1.40493080e+04 0.00000000e+00 3.64520100e+02]
 [0.00000000e+00 4.62615377e+03 2.16972632e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
7.203838700866708
dist_coefs
[[ 4.85915005e+01 -6.22411466e+03  1.29388851e+00 -4.01371903e-02
  -4.28539001e+03]]
Выборки 1,2,3,4(консоль 30А)

Directory
C:\Users\user\Downloads\many_chessboards1234
camera_matrix
[[1.90536220e+03 0.00000000e+00 3.22329890e+02]
 [0.00000000e+00 1.85628461e+03 1.37908247e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
8.526984802369714
dist_coefs
[[ 2.54249405e+00 -1.60258922e+01 -1.14349634e-01 -4.89189274e-02
   5.32506387e+02]]



Выборки 1,2,3,4,5,6фильтрованная, MANYDIST(консоль 32A)

SAMPLE SIZE 582
69289.76914858818
Directory
C:\Users\user\Downloads\many_chessboards123456+MANYDIST
camera_matrix
[[2.39711275e+03 0.00000000e+00 4.65292518e+02]
 [0.00000000e+00 1.52827851e+03 1.25495541e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
rms
6.7292723030971695
dist_coefs
[[ 2.83170500e+00 -6.69211644e+00  7.91372408e-02  7.05249300e-02
   3.55635609e+02]]
100%|██████████| 1165/1165 [00:13<00:00, 83.50it/s] 


Выборки 1,4,MANYDIST(консоль 33А)

Прелварительно на 9
SAMPLE SIZE 9
0.31049537658691406
Directory
C:\Users\user\Downloads\many_chessboards123456+MANYDIST
camera_matrix
[[742.15527975   0.         487.93779988]
 [  0.         510.61077698 354.75405995]
 [  0.           0.           1.        ]]
rms
7.157225255072494
dist_coefs
[[ 0.26120188 -0.22463931  0.05507762  0.05006945  0.14481679]]


camera_matrix
[[ 1.47612124e+03  0.00000000e+00  3.09005307e+02]
 [ 0.00000000e+00  1.06518129e+03 -6.29354626e+01]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]
rms
5.753453428918903
dist_coefs
[[ 1.05995926 -1.21305323 -0.25346465 -0.04999029  1.28114937]]
'''