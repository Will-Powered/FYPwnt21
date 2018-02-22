# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:04:05 2018

@author: wnt21
"""

import cv2
import numpy as np
import timeit

#load in image
img = cv2.imread('humanedit.jpg' )
time1 = timeit.default_timer()
#img = cv2.resize(img, (192, 108))
rows, cols = np.shape(img)[0:2]

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
val = hsv_image[:,:,2].astype(np.float64)

val2 = np.zeros((rows-2, cols-2))
cellsR = int(rows/16)
cellsC =     int(cols/16)
cells = np.zeros((cellsR, cellsC, 256))

for i in range(rows-2):
    for j in range(cols-2):
        temp = ''
        for k in range(-1, 2):
            for l in range(-1, 2):
                if (val[i+1+k,j+1+l]-val[i+1,j+1]) < 0:
                    temp = temp+'0'
                elif (k == 0) and (l==0):
                    temp = temp
                else:
                    temp = temp+'1'
        val2[i,j] = int(temp,2)
        if i < cellsR*16:
            cells[int(i/16),int(j/16), int(temp,2)] += 1

gMap = np.zeros((cellsR,cellsC))
gCell = cells[:,:,0]
gMap[gCell < 10] = 1
kernel = np.ones((3,3),np.uint8)
gMap = cv2.morphologyEx(gMap, cv2.MORPH_OPEN, kernel)

box =  np.nonzero(gMap)
target = img[np.min(box[0])*16:np.max(box[0])*16, np.min(box[1])*16:np.max(box[1])*16]

time2 = timeit.default_timer()

totaltime = time2-time1
print(totaltime)
cv2.imshow('img', gMap)
cv2.imshow('img2', target)

cv2.waitKey(0)
cv2.destroyAllWindows()
