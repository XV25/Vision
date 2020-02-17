import cv2
   
i=1;
cap = cv2.VideoCapture(0) 
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q'):
        break
    if key == ord('s'):
        nom='Data/Rot/image'+str(i)+'.png'
        cv2.imwrite(nom,frame)
        i=i+1;
   
cap.release()
cv2.destroyAllWindows()
