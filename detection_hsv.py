import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Bleu
    lower_blue = np.array([91,90,110])
    upper_blue = np.array([110,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #[1:]
    T = 0
    L = []
    cx,cy = 0,0
    for i in range (len(contours)):
        cnt = contours[i]
        M = cv.moments(cnt)
        if M['m00'] > 400:
            cx = int(M['m10']/(M['m00']+1*10**-5))
            cy = int(M['m01']/(M['m00']+1*10**-5))
            L.append([cx,cy])
            cv.circle(res,(cx,cy), 4, (0,255,100), -1) 
            break
    
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
