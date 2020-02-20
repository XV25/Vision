#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:27:15 2020

@author: ehnla
"""

import cv2
from detect_color_hsv import *
from check_detection_values import check_all_color

import numpy as np
import matplotlib.pylab as plt


plt.close('all')

import time

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
        c,d = good_old[0].ravel()
        mask = cv2.line(mask, (a2,b2),(c,d), color[5].tolist(), 2)
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(img_C,(a,b),5,color[i].tolist(),-1)
        
    return(img_C,mask,good_new)
    

def get_from_vid(name_vid,detect_colors =[],colors_values = {}):
    
    
    Mtime_dc = 0
    Mtime_LK = 0
    
    color = np.random.randint(0,255,(100,3))

    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    
    vidcap = cv2.VideoCapture(name_vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640,  480))
    k = 0
    success,old_image = vidcap.read()
    
    if colors_values == {}:
        if detect_colors == []:
            print("Detecting default colors\n")
            detect_colors =  ['R','V','B','O','J']
        colors_values = check_all_color(old_image, detect_colors)  
    
    M = np.zeros((len(colors_values),1,2), np.float32)
    i = 0
    
    for tcolor in colors_values:
        print("Detect color : ", tcolor)
        lower = colors_values[tcolor][0:3]
        upper = colors_values[tcolor][3:6]
        k1 = colors_values[tcolor][6]
        L = detect_color_vid2(old_image, lower, upper, k1)
        M[i,:,:] = np.array(L)
        i+=1
    
    print(M.shape)
    # mask = np.zeros_like(old_image)
    # good_old2 = np.zeros((1,2,1))
    p0 = M
    
    old_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_image)


    # upgrade : si distance entre deux points dentique trop grande 
    # --> risque de perte de point
    # --> réutiliser détection des couleurs
    
    
    while success:
        success,frame = vidcap.read()
        if (success):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
            t0 = time.time()
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            ftime = time.time()
            if (ftime - t0 > Mtime_LK):
                Mtime_LK = ftime - t0
                
            # Select good points
            
            good_new = p1[st==1]
            good_old = p0
            
            # if there is an error in the detection of pixels --> retry to detect the barycenters of the objects
            
            if good_new.shape[0]!= len(colors_values):
                    t0 = time.time()
                    #print("Lost pixel!")
                    M = np.zeros((len(colors_values),1,2), np.float32)
                    i = 0
                    for tcolor in colors_values:
                        lower = colors_values[tcolor][0:3]
                        upper = colors_values[tcolor][3:6]
                        k1 = colors_values[tcolor][6]
                        L = detect_color_vid2(old_image, lower, upper, k1)
                        M[i,:,:] = np.array(L)
                        i+=1
                    good_new = M
                    ftime = time.time()
                    if (ftime - t0 > Mtime_dc):
                        Mtime_dc = ftime - t0
                        

        # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
    
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    
        # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        
        
    out.release()
    cv2.destroyAllWindows()
    return(colors_values,Mtime_LK,Mtime_dc)

if __name__ == "__main__":
    img1  = cv2.imread("Data/Trans/image1.png")     # Lecture image en couleurs BGR
    img2 = cv2.imread("Data/Trans/image2.png")     # Lecture image en couleurs BGR
    
    #get_from_pic(img1,img2)
    
    S = {'R': np.array([149, 100,  48, 255, 255, 255,   0]),
      'V': np.array([ 45,  82,   0, 100, 255, 255,   0])}
    
    
    # S nathan
    
 #    S = {'R': np.array([126, 131, 200, 176, 255, 255,   2]),
 # 'V': np.array([ 65,  66,   0, 100, 165,  47,   2]),
 # 'B': np.array([100,  78,   0, 122, 180,  69,   2]),
 # 'O': np.array([  0, 128, 241,  24, 201, 255,   3]),
 # 'J': np.array([ 24, 129, 231,  52, 252, 255,   2])}
     
    gn,Mtime_LK,Mtime_dc = get_from_vid('output.avi',colors_values = S)
    
    print("Max time of color detection : ", Mtime_dc)
    print("Max time of LK detection : ", Mtime_LK)    
    
    
