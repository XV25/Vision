#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:27:15 2020

@author: ehnla
"""

import cv2
from detect_color_auto import *
import numpy as np
import matplotlib.pylab as plt


plt.close('all')

def get_from_pic(img1, img2, mask, color2, good_old2):
    
    #name1 = "Data/Trans/image1.png" # 0-255 / 0-15 / 181-230
    #img_C  = cv2.imread(name1)     # Lecture image en couleurs BGR
    
    img_C = img1
    
    img_C = cv2.resize(img_C,(640,480))
    
    old_gray = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)
    
    color = np.random.randint(0,255,(100,3))

    L = detect_color_vid(img_C, color2 )
    
    M = np.zeros((1,1,2), np.float32)
    
    M[0,:,:] = np.array(L)
    
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    p0 = M
    
    #fname  = "Data/Trans/image2.png" # 0-255 / 0-15 / 181-230
    #img_C  = cv2.imread(name2)     # Lecture image en couleurs BGR
    
    img_C = img2
    
    img_C = cv2.resize(img_C,(640,480))
    
    frame_gray = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)
    
    
    lk_params = dict( winSize  = (120,120),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    good_new = p1
    good_old = p0
    
    # draw the tracks
    if (good_old2[0,0,0] != 0):
        (a2,b2) = good_old2[0].ravel()
    
        frame = cv2.circle(img_C,(a2,b2),5,color[0].tolist(),-1)
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(img_C,(a,b),5,color[i].tolist(),-1)
        
    return(img_C,mask,good_new)
    

def get_from_vid(name_vid):
    vidcap = cv2.VideoCapture(name_vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640,  480))
    k = 0
    color = 'J'
    success,old_image = vidcap.read()
    mask = np.zeros_like(old_image)
    good_old2 = np.zeros((1,2,1))
    while success:
        success,new_image = vidcap.read()
        if success:
            imgC,mask,gn = get_from_pic(old_image, new_image, mask ,color, good_old2)
            new_image = cv2.add(imgC,mask)
            out.write(new_image)


            #cv2.imshow('frame %i'%k,new_image)
            k+=1
            old_image = new_image
            good_old2 = gn
            print(good_old2[0,0,0])
        if (k > 120):
            success = False
    out.release()
    cv2.destroyAllWindows()
    return(gn)

if __name__ == "__main__":
    img1  = cv2.imread("Data/Trans/image1.png")     # Lecture image en couleurs BGR
    img2 = cv2.imread("Data/Trans/image2.png")     # Lecture image en couleurs BGR
    
    #get_from_pic(img1,img2)
    gn = get_from_vid('output.avi')
    
    
    
