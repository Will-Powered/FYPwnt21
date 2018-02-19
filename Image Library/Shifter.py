import numpy as np
import cv2
import random

pic = cv2.imread('quick.jpg')
pic = cv2.resize(pic,(1920, 1080), interpolation = cv2.INTER_CUBIC)
map = cv2.imread('quickmap.png')
map = cv2.resize(map,(1920, 1080), interpolation = cv2.INTER_CUBIC)

rows,cols = pic.shape[0:2]

upShift = 200
rightShift = 200


def shift(img, up, right):
#wrapped translation
    wrap =  np.concatenate((img[up:,:,:], img[:up,:,:]))
    wrap =  np.concatenate((wrap[:,right:,:], wrap[:,:right,:]), axis=1)
    return wrap


def rot(img, angle):
#small rotation followed by zoom and crop to get rid of black areas
    rows,cols = img.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    temp = cv2.warpAffine(img,M,(cols,rows))
    res = cv2.resize(temp,None,fx=1.1, fy=1.1, interpolation = cv2.INTER_CUBIC)
    res= res[80:-80, 80:-80]
    res =cv2.resize(res,(cols, rows), interpolation = cv2.INTER_CUBIC)
    return res


def skew(img, indent):
#skew that leaves no black area
    pts1 = np.float32([[indent,0],[cols-indent,0],[0,rows],[cols,rows]])
    pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    skw = cv2.warpPerspective(img,M,(cols,rows))
    return skw

def flip(img, angle):
#small rotation followed by zoom and crop to get rid of black areas
    rows,cols = img.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    temp = cv2.warpAffine(img,M,(cols,rows))
    
    return temp


mod = pic
modmap = map

for i in range(10):
    if i%3 == 0:
        mod = pic
        modmap = map
    randUp = random.randint(-200,200)
    randAng = random.randint(-5,5)
    randSkw = random.randint(1, 100)
    randFlip = random.randint(0, 1)
    
    #mod = cv2.flip(mod, randFlip)
    #modmap = cv2.flip(modmap, randFlip)
    
    mod = flip(mod, 180*randFlip)
    modmap = flip(modmap, 180*randFlip)
    
    mod = shift(mod, randUp, randUp)
    modmap = shift(modmap, randUp, randUp)
    
    mod = rot(mod, randAng)
    modmap = rot(modmap, randAng)
    
    mod = skew(mod, randSkw)
    modmap = skew(modmap, randSkw)

    
    cv2.imwrite('modded' + str(i) + '.jpg', mod)
    cv2.imwrite('moddedmap' + str(i) + '.png', modmap)

# =============================================================================
# test = shift(pic, -200, -200)
# test =cv2.resize(test,(int(cols*0.8), int(rows*0.8)), interpolation = cv2.INTER_CUBIC)
# cv2.imshow('test', test)
# test2 = shift(pic, 200, 200)
# test2  =cv2.resize(test2,(int(cols*0.8), int(rows*0.8)), interpolation = cv2.INTER_CUBIC)
# cv2.imshow('test2', test2)
# =============================================================================

cv2.waitKey(0)
cv2.destroyAllWindows()