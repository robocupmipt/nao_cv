#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
#RG chromaticity
X_DIM = 640
Y_DIM = 480
class Line():
    def __init__(self,k, b, x_min,x_max):
        self.k,self.b=k,b
        self.xmin,self.xmax = int(x_min),int(x_max)
        y_extrs = [k*x_min+b,k*x_max+b]
        self.ymin,self.ymax = int(min(y_extrs)),int(max(y_extrs))
    def y(self,x, type_=int):
        return type_(self.k*x+self.b)
    
def turn_to_RG (img):
    (h, w, d) = img.shape
    
    norm = img.sum(2).astype(float)   
    norm [norm == 0] = 5
    turned = ((img.sum(2)/norm)*255).astype('uint8')    
    return turned

"""
def obtain_color_ratio_mask (img, components=(60,180,10), th=170, bl=61, use_rg =True):
    sh = img [:, :, 0].shape 
    if use_rg==False:
        rg = img
        smoothed = cv2.blur (rg, (bl, bl))
        needed = img.copy ()
        needed [:, :, 0] = np.full (sh, components [0])
        needed [:, :, 1] = np.full (sh, components [1])
        needed [:, :, 2] = np.full (sh, components [2]) 
    else:
        rg = turn_to_RG (img)
        smoothed = cv2.blur (rg, (bl, bl))
        raise Exception('needed is not defined yet while using rg')
    diff = cv2.absdiff (smoothed, needed) 
    dif = diff#dif is to be 2d while using rg  
    ret, res_mask = cv2.threshold (dif, th, 255, cv2.THRESH_BINARY_INV)    
    res_mask = cv2.morphologyEx (res_mask, cv2.MORPH_ERODE, np.ones ((int (bl), int (bl)), np.uint8))
    res_mask = cv2.morphologyEx (res_mask, cv2.MORPH_CLOSE, np.ones ((int (bl), int (bl)), np.uint8)) 
    res = cv2.bitwise_and (img, img, mask = res_mask)
    
    return res, res_mask
    #return diff
"""

def obtain_color_ratio_mask (img, components, th, bl, use_rg = False):
    sh = img [:, :, 0].shape
    
    rg = img
    
    if (use_rg == True):
        rg = turn_to_RG (img)
    
    smoothed = cv2.blur (rg, (bl, bl))
    
    needed = img.copy ()
    needed [:, :, 0] = np.full (sh, components [0])
    needed [:, :, 1] = np.full (sh, components [1])
    needed [:, :, 2] = np.full (sh, components [2])
    
    diff = cv2.absdiff (smoothed, needed)
    
    dif = diff [:, :, 0] + diff [:, :, 1] + diff [:, :, 2]
    
    ret, res_mask = cv2.threshold (dif, th, 255, cv2.THRESH_BINARY_INV)
    
    res_mask = cv2.morphologyEx (res_mask, cv2.MORPH_ERODE, np.ones ((int (bl), int (bl)), np.uint8))
    res_mask = cv2.morphologyEx (res_mask, cv2.MORPH_CLOSE, np.ones ((int (bl), int (bl)), np.uint8))
    
    res = cv2.bitwise_and (img, img, mask = res_mask)
    
    return res, res_mask

def erase_little_parts (mask, area_th, hei_th, wid_th):
    result = np.array (mask)
    output = cv2.connectedComponentsWithStats (mask, 8, cv2.CV_32S)
    labels       = output      [1]
    stats        = output      [2]
    sz           = stats.shape [0]    
    for label_num in range (0, sz - 1):
        if (stats [label_num, cv2.CC_STAT_AREA]   < area_th or
            stats [label_num, cv2.CC_STAT_WIDTH]  < wid_th  or
            stats [label_num, cv2.CC_STAT_HEIGHT] < hei_th):
            result [labels == label_num] = 0 
    return result
def are_close(lines1,lines2, max_closedist=10):
    '''
    Чтобы сэкономить время, мы измеряем только расстояние по оси y при том же самом x
    delta = x*(k1-k2)+(b1-b2)
    ищем min(abs(delta)) при x=min_x, x=max_x 
    или если delta= 0 при x между [min_x,max_x], То возвращаем 0
    '''
    min_x = max(lines1.xmin,lines2.xmin)
    max_x = min(lines1.xmax,lines2.xmax)
    k1 = lines1.k-lines2.k
    b1=lines1.b-lines2.b
    if min_x>max_x:
        return False
    close_delta1 = max(k1*min_x+b1, k1*max_x+b1)
    if ((close_delta1<max_closedist)):
        return True
    return False
        
def line_filter(lines):
    lines = sorted(lines, key = lambda z: z.xmin -z.xmax)
    to_delete=[]
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            if (j not in to_delete
                and are_close(lines[i],lines[j])):
                to_delete.append(j)
    to_delete=sorted(to_delete)            
    for ind in to_delete[::-1]:
        del lines[ind]
    return lines

def getLines (img):
    
    field, f_mask = obtain_color_ratio_mask (img, (60, 180, 10), 170, 61, use_rg = False)
    #field2, f2_mask = obtain_color_ratio_mask (img, (60, 180, 10), 170, 5, use_rg = False)
    field2, f2_mask = obtain_color_ratio_mask (img, (60, 180, 10), 170, 1, use_rg = False)
    f2_mask_inv = cv2.bitwise_not (f2_mask)
    field_cut = cv2.bitwise_and (field, field, mask = f2_mask_inv)  
    field_cut_resized = cv2.resize (field_cut, (X_DIM,Y_DIM))       
    gray = cv2.cvtColor(field_cut_resized,cv2.COLOR_BGR2GRAY)#I use only field_cut_resized
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny (blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 45  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 60  # maximum gap in pixels between connectable line segments
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    answer=[]
    if lines is not None:
        for line in lines:
            print(line)
            print(line.shape)
            x1,y1,x2,y2=line[0]
            k=(y2-y1)/(x2+0.0-x1)#to be float
            b=(y1*x2-x1*y2)/(x2+0.0-x1)#to be float
            x_min=min(x1,x2)
            x_max=max(x1,x2)
            answer.append(Line(k,b,x_min,x_max))
        answer = line_filter(answer)
    return answer