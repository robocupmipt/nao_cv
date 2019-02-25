# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:00:01 2019

@author: DK
"""


import numpy as np
import cv2
from line_detection import getLines
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
def log(image, lines,other_lines, image_name, thickness=1):
    colors=[(255,0,0),(0,255,0),(0,0,255)]
    for line in other_lines:
        cv2.line(image,
            (line.xmin,line.y(line.xmin)),
            (line.xmax,line.y(line.xmax)),(0,0,0),thickness)
    for (line,color) in zip(lines,colors):
        if line is not None:
            cv2.line(image,
            (line.xmin,line.y(line.xmin)),
            (line.xmax,line.y(line.xmax)),color,thickness)
            
    cv2.imwrite('edited'+image_name+'.jpg',image)
import os
os.chdir('C:\Users\DK\Downloads\images')
for image_name in os.listdir(os.getcwd()) :
    image = cv2.imread(image_name)
    other_lines = getLines(image)
    lines = processLines(other_lines)
    log(image,lines,other_lines,image_name)
