# -*- coding: utf-8 -*-

import os
import cv2
os.chdir('C:\\Users\\user\\Downloads\\many_chessboards7')
import _pickle as cPickle
calib_data = cPickle.load(open('calib_data.pkl','rb'))
images = [j for j in os.listdir() if '.jpg' in j]
mse, calib_matrix,dist_coefs, rvecs, tvecs = calib_data
cell_size=2.45#size of 1 cell, CALIBRATE TAKING IT INTO ACCOUNT
calib_matrix = calib_matrix
w,h=640,480
new_calibmatrix, roi=cv2.getOptimalNewCameraMatrix(calib_matrix, dist_coefs,(w,h),1,(w,h))
import cv2
import numpy as np
from tqdm import tqdm
s=1
def transform(image):
    dst = cv2.undistort(image, calib_matrix, dist_coefs, None, new_calibmatrix)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
def get_newcoords(x,y, rt_index, s=s):
    global rvecs,tvecs
    '''
    x, y -  координаты на dst
    
    s* pixel_coords = newcameramtx *(R|T) * coords_3d
    Т.к нам нужны coords_3d,а не coords_4d, мы убираем у newcameramtx*(R|T) последний столбей,
    делая эту матрицу размером не 3*4, а 3*3. Обозначим такую матрицу как newcameramtx*(R|T)_
    s*((newcameramtx*(R|T)_).INV())*pixel_coords = coords_3d
    '''
    R=cv2.Rodrigues(rvecs[rt_index])[0]
    T = tvecs[rt_index]
    RT = np.concatenate((R,T),axis=1)
    pixel_vector = np.array([x,y,1])
    matrix1 = np.matmul(new_calibmatrix, RT)[:3,:3]#Matrix1 is 3*4
    matrix2 = np.linalg.inv(matrix1)
    coords_3d = s*np.matmul(matrix2, pixel_vector)
    return coords_3d

#min_dists=[]
#for (rt_index, test_imagename) in tqdm(enumerate(images)):
#    test_image = cv2.imread(test_imagename)
#    '''
#    Transforming test image
#    '''
#    h,w=test_image.shape[:2]
#    new_calibmatrix, roi=cv2.getOptimalNewCameraMatrix(calib_matrix, dist_coefs,(w,h),1,(w,h))
#    dst = cv2.undistort(test_image, calib_matrix, dist_coefs, None, new_calibmatrix)
#    x,y,w,h = roi
#    dst = dst[y:y+h, x:x+w]
#    (found,corners)=cv2.findChessboardCorners(dst,(9,6),
#       flags =cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
#    new_coords=[]
#    for corner in corners:
#        coord = get_newcoords(corner[0][0],corner[0][1],rt_index)
#        new_coords.append(coord)
#    '''
#    Сейчас надо отмасштабировать найденные координаты - 
#    для этого смотрим для каждой координаты минимальную дистанцию до другой - 
#    дистанция должна быть равна cell_size(2.45 см)
#    '''
#    for coord in new_coords:
#        min_dist=999999999
#        for coord1 in new_coords:
#            if coord1[0]!=coord[0] or coord[1]!=coord1[1] or coord[2]!=coord1[2]:
#                min_dist=min(min_dist, (sum((coord-coord1)**2))**0.5)
#        min_dists.append(min_dist)



#avg_mindist=sum(min_dists)/len(min_dists)
#s1=cell_size/avg_mindist
#s=s*s1
#cv2.imwrite('calibresult.png',dst)
#image_name='ggfddgf'#Name of image we are testing on
image_name='gdf'
image_xy=[(0,1),(1,2),(2,3)]
real_xy=[(5,1),(5,2),(5,3)]
delta = 0
def dist(pair1,pair2):
    return ((pair1[0]-pair2[0])**2 + (pair1[1]-pair2[1])**2)**0.5
def edit_photo(image_name, rt_index=None):
    calib_img = cv2.imread(image_name,0)
    calib_img=transform(calib_img)
    avg_dists=[]
    for i in range(len(rvecs)):
        predicted_xy = [get_newcoords(x_,y_,i) for (x_,y_) in image_xy]
        distances =[dist(predicted_xy[i],real_xy[i]) for i in range(len(real_xy))]
        avg_dists.append(sum(distances)/len(distances))
    return np.argmax(avg_dists),avg_dists   
edit_photo(image_name)
            
    
    