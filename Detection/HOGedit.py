# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:38:10 2018

@author: wnt21
"""

import cv2
import numpy as np
import timeit

#load in image
img = cv2.imread('A1920.JPG')
time1 = timeit.default_timer()
rows, cols = np.shape(img)[0:2]

#downsample
img = cv2.resize(img, (int(cols*0.25), int(rows*0.25)))
#img = cv2.blur(img, (5,5))


# x and y sobel on sep channels
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

#sums each channel together
gX = np.sum(gx, axis=2)
gY = np.sum(gy, axis=2)

#finds gradient magnitude
#gMag = np.sqrt(np.add(np.power(gX,2), np.power(gY,2)))
gMag = np.linalg.norm([gX, gY], axis = 0)

#finds gradient angle and normalises to 0<theta<180
theta = np.arctan2(gY, gX)
theta[theta < 0] = np.add(theta[theta < 0], np.pi)
theta = np.divide(theta, np.pi/180)
theta[theta == 180] = 0


#finds the lower bin that each gradient will contribute to
lowerBin = (np.divide(theta, 20)).astype(np.uint)

#each cell is 8x8 and votes in a histogram with 9 bins
cellsW = int(cols/32)
cellsH = int(rows/32)
angBin = np.zeros((cellsH, cellsW, 9))

#temp2 = np.zeros((np.shape(theta)[0], np.shape(theta)[1], 9))
#for i in range(8):
#    temp2[:,:,i] = np.multiply(np.divide(np.add(-(np.subtract(theta, i*20)), 20),20),gMag)

kernel = np.ones((3,3),np.uint8)
kernel2 =np.ones((3,3),np.uint8)
gMag = cv2.morphologyEx(gMag, cv2.MORPH_CLOSE, kernel2)
gMag = cv2.morphologyEx(gMag, cv2.MORPH_OPEN, kernel)

box =  np.where(gMag < 50)

target = img[np.min(box[0]):np.max(box[0]), np.min(box[1]):np.max(box[1])]
target = cv2.resize(target, (50,50)) 
cellsW = int(target.shape[1]/8)
cellsH = int(target.shape[0]/8)
angBin = np.zeros((cellsH, cellsW, 9))

for x in range(cellsW):
    for y in range(cellsH):
        
        tlowerBin = lowerBin[y:y+8, x:x+8]
        ttheta = theta[y:y+8, x:x+8]
        tgMag = gMag[y:y+8, x:x+8]
        for i in range(8):
            temp = np.multiply(np.divide(np.add(-(np.subtract(ttheta[tlowerBin == i], i*20)), 20),20),tgMag[tlowerBin == i])
            angBin[y,x,i] = np.sum(temp)
            angBin[y,x,i+1] = np.sum(np.subtract(tgMag[tlowerBin == i], temp))
    
        temp = np.multiply(np.divide(np.add(-(np.subtract(ttheta[tlowerBin == 8], 8*20)), 20),20),tgMag[tlowerBin == 8])
        angBin[y,x,8] = np.sum(temp)
        angBin[y,x,0] = np.sum(np.subtract(tgMag[tlowerBin == 8], temp))


blocks = np.zeros((cellsH-1, cellsW-1, 36))
for i in range(cellsH-1):
    for j in range(cellsW-1):
        blocks[i,j,:] = np.reshape(angBin[i:i+2, j:j+2, :], (36))

time2 = timeit.default_timer()

timetotal = time2 - time1

print(timetotal)



cv2.imshow('img', target)

cv2.waitKey(0)
cv2.destroyAllWindows()