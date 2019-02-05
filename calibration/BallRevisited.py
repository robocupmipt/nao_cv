# -*- coding: utf-8 -*-
"""
Правильно ли я сделал? Ведь если бы я взял какой-нибудь другой rvecs и tvecs, функция get_newcoords уже выдала бы другие значения.
"""

import _pickle as cPickle
calib_data = cPickle.load(open('calib_data.pkl','rb'))
mse, calib_matrix,dist_coefs, rvecs, tvecs = calib_data
import cv2
import numpy as np
total_rvecs = np.concatenate(rvecs, axis=1)
R = cv2.Rodrigues(rvecs[0])[0]#How should I concatenate several ones?
T = tvecs[0]
RT = np.concatenate((R,T),axis=1)
TEST_IMAGENAME = 'test_image0.jpg'
test_image = cv2.imread(TEST_IMAGENAME)
'''
Transforming test image
'''
h,w=test_image.shape[:2]
new_calibmatrix, roi=cv2.getOptimalNewCameraMatrix(calib_matrix, dist_coefs,(w,h),1,(w,h))
dst = cv2.undistort(test_image, calib_matrix, dist_coefs, None, new_calibmatrix)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
def get_newcoords(x,y):
    '''
    x, y -  координаты на dst
    
    s* pixel_coords = newcameramtx *(R|T) * coords_3d
    Т.к нам нужны coords_3d,а не coords_4d, мы убираем у newcameramtx*(R|T) последний столбей,
    делая эту матрицу размером не 3*4, а 3*3. Обозначим такую матрицу как newcameramtx*(R|T)_
    s*((newcameramtx*(R|T)_).INV())*pixel_coords = coords_3d
    '''
    s = 91.3811628408346#Поиск этого коэффициента будет приведен ниже
    pixel_vector = np.array([x,y,1])
    matrix1 = np.matmul(new_calibmatrix, RT)[:3,:3]#Matrix1 is 3*4
    matrix2 = np.linalg.inv(matrix1)
    coords_3d = s*np.matmul(matrix2, pixel_vector)
    return coords_3d
(found,corners)=cv2.findChessboardCorners(dst,(9,6),
   flags =cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
new_coords=[]
for corner in corners:
    coord = get_newcoords(corner[0][0],corner[0][1])
    new_coords.append(coord)
'''
Сейчас надо отмасштабировать найденные координаты - 
для этого смотрим для каждой координаты минимальную дистанцию до другой - 
дистанция должна быть равна cell_size(2.45 см)
'''
min_dists=[]
for coord in new_coords:
    min_dist=999999999
    for coord1 in new_coords:
        if coord1[0]!=coord[0] or coord[1]!=coord1[1] or coord[2]!=coord1[2]:
            min_dist=min(min_dist, (sum((coord-coord1)**2))**0.5)
    min_dists.append(min_dist)
'''
Но минимальные дистанции разные - они отличаются до 30%. Неужели cv2.undistort убирал не все искажения?
'''
avg_mindist=sum(min_dists)/len(min_dists)
s1=2.45/avg_mindist
cv2.imwrite('calibresult.png',dst)