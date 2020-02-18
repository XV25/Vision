import numpy as np
import cv2

#cap = cv2.VideoCapture('output.avi')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


#cap = cv2.VideoCapture('nathan.avi')
cap = cv2.VideoCapture(0)
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
    
lower_blue = np.array([164,0,100])
upper_blue = np.array([198,255,177])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('res',mask)      
kernelf = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
kernelo = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelf)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelo)
# Bitwise-AND mask and original image
#res = cv2.bitwise_and(frame,frame, mask= mask)
res = old_frame.copy()
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #[1:]
cx,cy = 0,0

Mmax = 0
cnt_max = None

for i in range (len(contours)):
    cnt = contours[i]
    M = cv2.moments(cnt)
    if M['m00'] > Mmax:
        Mmax = M['m00']
        cnt_max = cnt
 
M = cv2.moments(cnt_max)       
cx = int(M['m10']/(M['m00']+1*10**-5))
cy = int(M['m01']/(M['m00']+1*10**-5))
cv2.circle(mask,(cx,cy), 4, (0,255,100), -1) 
        

cv2.imshow('res2',mask)      
M = np.zeros((1,1,2), np.float32)
M[0,:,:] = np.array([cx,cy])
#M[0,:,:] = np.array([300,300])
p0 = M


print(p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
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

cv2.destroyAllWindows()
cap.release()
