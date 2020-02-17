##############################################################################
import numpy as np  # module pour la manipulation de matrice

import pylab as plt # module pour affichage des données
#from matplotlib import pyplot as plt    # Module image propre à python 
#from scipy.ndimage import label, generate_binary_structure

import cv2          # module pour la manipulation d'image via OpenCV

###############################################################################
# Lecture d'une image & information sur l'image


# sélection de toutes les couleurs fluos : 
# S : 175 - 255
# Pour bleu : 
# H : 90 - 160
# Pour vert : 
# H : 60 - 95
# Pour jaune : 
# H : 25 - 50
# Pour orange : 
# H : 0 - 20
# Pour rose : 
# H : 160 - 175

# Sélection B : 
#

from threading import Thread
import cv2
#from imutils.video.pivideostream import PiVideoStream

import time


###############################################################################
# Application filtrage d'une image couleur
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html puis voir les descriptions des méthodes (liens)


###############################################################################
# 1. Réhaussement de contraste d'une image couleur
def Egalisation_HSL_col(img_BGR):
    img_HSV = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV) # Image BGR --> HSL
    h,s,v   = cv2.split(img_HSV)                      # Extraction des 3 plans HSL notamment value v
    h_egal = cv2.equalizeHist(h)
    s_egal  = cv2.equalizeHist(s)                     # Egalisation histogramme sur s
    v_egal  = cv2.equalizeHist(v)                     # Egalisation histogramme sur v
    
    img_egal= img_HSV.copy()                          # Copie de l'image HSL
    
    return img_egal


def detect_color_vid(image,color): 
    
    #Etape 1 : égalisation  HSL
    # inutile pour le moment

    img_egalisation = Egalisation_HSL_col(img_C)
    #    cv2.namedWindow("Ega", cv2.WINDOW_NORMAL) 
#    cv2.imshow("Ega", img_egalisation)

    # Etape 2 : filtrage 
    # inutile pour le moment
 #   taille   = 7

  #  img_Gaus = img_egalisation.copy()
    
#    
#    cv2.namedWindow("Gaussian", cv2.WINDOW_NORMAL) 
#    cv2.imshow("Gaussian", img_Gaus)

#img_egalisation = img_Gaus
# Init des seuils 

    # imgHSL = cv2.cvtColor(img_C,cv2.COLOR_BGR2HLS)
    imgHSL = img_egalisation
    
    if color == "B":
        Hmin = 91
        Hmax = 119
        Smin = 90
        Smax = 255
        Vmin = 110
        Vmax = 255
        k1 =1
        kf = 2*k1 + 1 
        k2 =1
        ko = 2*k2 + 1 
    
    elif color == "V":
        Hmin = 49
        Hmax = 101
        Smin = 90
        Smax = 255
        Vmin = 110
        Vmax = 255
        k1 =1
        kf = 2*k1 + 1 
        k2 =1
        ko = 2*k2 + 1 


    elif color == "J":
        Hmin = 21
        Hmax = 69
        Smin = 90
        Smax = 255
        Vmin = 110
        Vmax = 255
        k1 =1
        kf = 2*k1 + 1 
        k2 =1
        ko = 2*k2 + 1 


    elif color == "O":
        Hmin = 0
        Hmax = 24
        Smin = 90
        Smax = 255
        Vmin = 110
        Vmax = 255
        k1 =1
        kf = 2*k1 + 1 
        k2 =1
        ko = 2*k2 + 1 
    
    elif color == "R":
        Hmin = 145
        Hmax = 177
        Smin = 90
        Smax = 255
        Vmin = 110
        Vmax = 255
        k1 =1
        kf = 2*k1 + 1 
        k2 =1
        ko = 2*k2 + 1 

    lower = np.array([Hmin,Smin,Vmin])
    upper = np.array([Hmax,Smax,Vmax])
        
    img_bin = cv2.inRange(imgHSL,lower,upper)

    if k1 == -1:
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #[1:]
    else :
        kernelf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kf, kf))
        kernelo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        et1 = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernelf)
        et2 = cv2.morphologyEx(et1, cv2.MORPH_OPEN, kernelo)
        contours, hierarchy = cv2.findContours(et2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #[1:]
    #cv2.drawContours(imgfinal, contours, -1, (255,255,0), 1, cv2.LINE_8, hierarchy)
    #cv2.imshow("image contours",imgfinal)

    T = 0
    L = []
    cx,cy = 0,0
    for i in range (len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        if M['m00'] > 400:
            cx = int(M['m10']/(M['m00']+1*10**-5))
            cy = int(M['m01']/(M['m00']+1*10**-5))
            L.append([cx,cy])
            #print(M['m00'])
            cv2.circle(img_C,(cx,cy), 4, (0,255,100), -1) 
            #(x,y),(Ma,ma),angle =  cv2.fitEllipse(cnt)
            #angle : angle de rotation de l'ellipse.
          
            #area = cv2.contourArea(cnt)
            #x,y,w,h = cv2.boundingRect(cnt)
            #rect_area = w*h
            #extent = float(area)/rect_area
            break
        
    cv2.imshow('Mon masque',et2)
    cv2.imshow("image centroides",img_C)
    return([cy,cx])
    
if __name__ == "__main__":
    
    plt.close('all')
    fname  = "image1.png" # 0-255 / 0-15 / 181-230
    img_C  = cv2.imread(fname)     # Lecture image en couleurs BGR
    
    img_C = cv2.resize(img_C,(640,480))
    
     
    # loop over some frames...this time using the threaded stream
    
    	# grab the frame ffrom the threaded video stream and resize it
    	# to have a maximum width of 400 pixels
    
    # Affichage image via opencv --> affichage format BGR
    plt.figure(1)
    plt.imshow(img_C)
    L = detect_color_vid(img_C,'B')
    print(L)
