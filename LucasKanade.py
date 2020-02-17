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

fname  = "Data/Trans/image1.png" # 0-255 / 0-15 / 181-230
img_C  = cv2.imread(fname)     # Lecture image en couleurs BGR

img_C = cv2.resize(img_C,(640,480))

old_gray = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(img_C)

color = np.random.randint(0,255,(100,3))

L = detect_color_vid(img_C,'B')

M = np.zeros((1,1,2), np.float32)

M[0,:,:] = np.array(L)


feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

p0 = M

fname  = "Data/Trans/image2.png" # 0-255 / 0-15 / 181-230
img_C  = cv2.imread(fname)     # Lecture image en couleurs BGR

img_C = cv2.resize(img_C,(640,480))

frame_gray = cv2.cvtColor(img_C, cv2.COLOR_BGR2GRAY)


lk_params = dict( winSize  = (100,100),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))


# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

good_new = p1[st==1]
good_old = p0[st==1]

    # draw the tracks
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv2.circle(img_C,(a,b),5,color[i].tolist(),-1)
img = cv2.add(img_C,mask)
cv2.imshow('frame',img)