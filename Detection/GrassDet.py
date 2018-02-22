# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:10:18 2018

@author: wnt21
"""

import cv2
import numpy as np
import timeit
#import matplotlib.pyplot as plt


time1 = timeit.default_timer()
#load in image
img2 = cv2.imread('A1920.JPG')
imH, imW = img2.shape[0:2]
#downsize image
img = cv2.resize(img2, (int(imW/4), int(imH/4)))
rows, cols = np.shape(img)[0:2]
img = cv2.blur(img, (15,15))

#convert image to hsv and bitplane slice
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#hsv_image = (np.round(hsv_image/15)*15).astype(np.uint8)
hsv_image = (np.multiply(np.round(np.divide(hsv_image, 15)),15)).astype(np.uint8)


#creates mask that is 0 where hue is green
mask = np.ones_like(hsv_image[:,:,0])

mask[hsv_image[:,:,0] == 30] = 0
mask[hsv_image[:,:,0] == 45] = 0
mask[hsv_image[:,:,0] == 60] = 0
mask[hsv_image[:,:,0] == 75] = 0

mask = np.repeat(mask.reshape((rows, cols, 1)), 3, axis = 2)

#multiplies image with mask to delete green areas
final = np.multiply(hsv_image, mask).astype(np.uint8)


#morphological open to delete small areas
kernel = np.ones((9,9),np.uint8)
final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)

#finds bounds of middle box and crops original image
box =  np.nonzero(final)
target = img2[np.min(box[0])*4:np.max(box[0])*4, np.min(box[1])*4:np.max(box[1])*4]
time2 = timeit.default_timer()

totaltime = time2-time1
print(totaltime)
#plt.plot(hist)
#plt.show()

cv2.imshow('img', target)
#cv2.imshow('final', final)

cv2.waitKey(0)
cv2.destroyAllWindows()