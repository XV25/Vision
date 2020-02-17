# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:24:55 2019

@author: ehnla
"""

# -*- coding: utf-8 -*-

###############################################################################
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


fname  = "Data/Trans/image1.png" # 0-255 / 0-15 / 181-230
img_C  = cv2.imread(fname)     # Lecture image en couleurs BGR

img_C = cv2.resize(img_C,(640,480))

 
# loop over some frames...this time using the threaded stream

	# grab the frame ffrom the threaded video stream and resize it
	# to have a maximum width of 400 pixels

# Affichage image via opencv --> affichage format BGR
plt.figure(1)
plt.imshow(img_C)

cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL) 
cv2.imshow("mon image BGR", img_C)

###############################################################################
# Application filtrage d'une image couleur
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html puis voir les descriptions des méthodes (liens)


###############################################################################
# 1. Réhaussement de contraste d'une image couleur
def Egalisation_BGR(img_BGR):
    img_HSV = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HLS) # Image BGR --> HSV
    h,l,s   = cv2.split(img_HSV)                      # Extraction des 3 plans HSV notamment value v
    h_egal = cv2.equalizeHist(h)
    s_egal  = cv2.equalizeHist(s)                     # Egalisation histogramme sur s
    l_egal  = cv2.equalizeHist(l)                     # Egalisation histogramme sur v
    
    img_egal= img_HSV.copy()                          # Copie de l'image HSV
   # img_egal[:,:,0] = h_egal
  #  img_egal[:,:,2] = s_egal                          # Modification du plan s
    img_egal[:,:,1] = l_egal                          # Modification du plan v
    
    
    # Uniquement sur L, pt sur S, pas sur H
    #img_result      = cv2.cvtColor(img_egal,cv2.COLOR_HLS2BGR) # Image HSV --> BGR 
    
    return img_egal

# 2. Mise en oeuvre galisation HSV
img_egalisation = Egalisation_BGR(img_C)

#Filtre gaussien si image très bruité : pertinence ici? Pt homogénéiser rond, lisser zones grises plus blanches?

# taille   = 7
#img_Mean = cv2.blur(img_C,(taille,taille))           # filtrage moyenneur - Traitement marginal
#img_Gaus = cv2.GaussianBlur(img_egalisation,(taille,taille),0) # filtrage Gaussien - Traitement marginal
#img_median = cv2.medianBlur(img_C, taille)           # filtrage median (non linéaire)
# img_Gaus = img_egalisation.copy()
# cv2.namedWindow("Ega", cv2.WINDOW_NORMAL) 
# cv2.imshow("Ega", img_egalisation)

# cv2.namedWindow("Gaussian", cv2.WINDOW_NORMAL) 
# cv2.imshow("Gaussian", img_Gaus)
#cv2.namedWindow("Filter2D", cv2.WINDOW_NORMAL) 
#cv2.imshow("Filter2D", img_fil)  

img_Gaus = img_egalisation
# Init des seuils 
h_min, h_max = 40, 80 # intervalle Teinte (H) --> (40,80) pour le vert (110,130) pour le bleu (140,180)
s_min, s_max = 0, 255 # intervalle Saturation (S)
v_min, v_max = 0, 255 # intervalle Valeurs (V)
k1  = 0
imgHSV = img_Gaus.copy()# cv2.cvtColor(img_Gaus,cv2.COLOR_BGR2HLS)

def binarizeHSV(pos):
    Hmin = cv2.getTrackbarPos('Hmin','Mon masque')
    Hmax = cv2.getTrackbarPos('Hmax','Mon masque')
    Smin = cv2.getTrackbarPos('Smin','Mon masque')
    Smax = cv2.getTrackbarPos('Smax','Mon masque')
    Lmin = cv2.getTrackbarPos('Lmin','Mon masque')
    Lmax = cv2.getTrackbarPos('Lmax','Mon masque')
    k1 = cv2.getTrackbarPos('Taille noyaux','Mon masque')
    k = 2*k1 + 1 
    print(k)
    cercle_detect = 0
   # print(k)
    lower = np.array([Hmin,Lmin,Smin])
    upper = np.array([Hmax,Lmax,Smax])
   # print(lower)
    img_bin = cv2.inRange(imgHSV,lower,upper)
    img_calc = cv2.bitwise_and(imgHSV,imgHSV,mask = img_bin)
    
    
    #img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #♣cv2.namedWindow('Mon masque',cv2.WINDOW_NORMAL)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    et1 = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    img3 = cv2.bitwise_and(img_calc,img_calc,mask=et1)
#    img_bin2 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
#    et2 = cv2.morphologyEx(img_bin2, cv2.MORPH_OPEN, kernel)
#    img3 = cv2.bitwise_and(img5,img5,mask=et2)
    cv2.imshow('Ap opening',et1)
    bin_post = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    
    imgfinal = img_egalisation.copy()
    
    contours, hierarchy = cv2.findContours(bin_post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#[1:]
    T = False
    for i in range (len(contours)):
        cnt = contours[i]
        M = cv2.moments(cnt)
        if M['m00'] > 400:
            cx = int(M['m10']/(M['m00']+1*10**-5))
            cy = int(M['m01']/(M['m00']+1*10**-5))
            print(M['m00'])
            cv2.circle(imgfinal,(cx,cy), 4, (0,0,255), -1) 
            (x,y),(Ma,ma),angle =  cv2.fitEllipse(cnt)
            #angle : angle de rotation de l'ellipse.
          #  cv2.circle(imgfinal,(int(x),int(y)), 4, (0,255,255), -1)
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            rect_area = w*h
            extent = float(area)/rect_area
            T = True
            break

    cv2.imshow("image centroides",imgfinal)
    

    cv2.imshow('Mon masque',img_bin)


#cv2.imshow('Mon masque',img2)
    
#cv2.imshow('Mon Résultat',img_calc)
cv2.namedWindow('Mon masque',cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hmin','Mon masque',0,255,binarizeHSV)
cv2.createTrackbar('Hmax','Mon masque',cv2.getTrackbarPos('Hmin','Mon masque'),255,binarizeHSV)
cv2.createTrackbar('Smin','Mon masque',0,255,binarizeHSV)
cv2.createTrackbar('Smax','Mon masque',cv2.getTrackbarPos('Smin','Mon masque'),255,binarizeHSV)
cv2.createTrackbar('Lmin','Mon masque',0,255,binarizeHSV)
cv2.createTrackbar('Lmax','Mon masque',cv2.getTrackbarPos('Lmin','Mon masque'),255,binarizeHSV)
cv2.createTrackbar('Taille noyaux','Mon masque',0,18,binarizeHSV)
# Test
binarizeHSV(0)

# Creation des barres de défilement

##### FIN
cv2.waitKey(0)                     
cv2.destroyAllWindows()

#    cv2.drawContours(imgfinal, contours, -1, (255,255,0), 1, cv2.LINE_8, hierarchy)
#    cv2.imshow("image contours",imgfinal)
#   # cv2.imwrite("composante_connexes.jpg",imgfinal)
#    for i in range (len(contours)):
#        cnt = contours[i]
#        M = cv2.moments(cnt)
#        cx = int(M['m10']/(M['m00']+1*10**-5))
#        cy = int(M['m01']/(M['m00']+1*10**-5))
#        print(cx,cy)
    
#        cv2.circle(imgfinal,(cx,cy), 4, (0,0,255), -1)