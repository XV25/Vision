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


 # 0-255 / 0-15 / 181-230


###############################################################################
# Application filtrage d'une image couleur
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html puis voir les descriptions des méthodes (liens)


###############################################################################
# 1. Réhaussement de contraste d'une image couleur
def Egalisation_BGR(img_BGR):
    img_HSV = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV) # Image BGR --> HSV
    h,s,v   = cv2.split(img_HSV)                      # Extraction des 3 plans HSV notamment value v
    h_egal = cv2.equalizeHist(h)
    s_egal  = cv2.equalizeHist(s)                     # Egalisation histogramme sur s
    v_egal  = cv2.equalizeHist(v)                     # Egalisation histogramme sur v
    
    img_egal= img_HSV.copy()                          # Copie de l'image HSV
   # img_egal[:,:,0] = h_egal
    img_egal[:,:,2] = v_egal                          # Modification du plan s
    #img_egal[:,:,1] = s_egal                          # Modification du plan v
    
    
    # Uniquement sur L, pt sur S, pas sur H
    #img_result      = cv2.cvtColor(img_egal,cv2.COLOR_HLS2BGR) # Image HSV --> BGR 
    
    return img_egal

# 2. Mise en oeuvre galisation HSV

#cv2.imshow('Mon masque',img2)

def check_color(img_C):    # Lecture image en couleurs BGR

    img_C = cv2.resize(img_C,(640,480))

 
# loop over some frames...this time using the threaded stream

	# grab the frame ffrom the threaded video stream and resize it
	# to have a maximum width of 400 pixels

# Affichage image via opencv --> affichage format BGR
    plt.figure(1)
    plt.imshow(img_C)

    cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL) 
    cv2.imshow("mon image BGR", img_C)
#cv2.imshow('Mon Résultat',img_calc)
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
        T = 0
        L = []
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
        cv2.circle(img_C,(cx,cy), 4, (0,255,100), -1) 
    
    
        cv2.imshow("image centroides",imgfinal)
        
    
        cv2.imshow('Mon masque',img_bin)
        
        return(None)

    
    
    
    cv2.namedWindow('Mon masque',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Hmin','Mon masque',0,360,binarizeHSV)
    cv2.createTrackbar('Hmax','Mon masque',cv2.getTrackbarPos('Hmin','Mon masque'),360,binarizeHSV)
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
 


def check_all_color(img_C,colors):    # Lecture image en couleurs BGR

    colors_values = {}
    
    img_C = cv2.resize(img_C,(640,480))

 
# loop over some frames...this time using the threaded stream

	# grab the frame ffrom the threaded video stream and resize it
	# to have a maximum width of 400 pixels

# Affichage image via opencv --> affichage format BGR
    
    for color in colors : 
    
        cv2.namedWindow("mon image BGR", cv2.WINDOW_NORMAL) 
        cv2.imshow("mon image BGR", img_C)
    #cv2.imshow('Mon Résultat',img_calc)
        img_egalisation = Egalisation_BGR(img_C)
        
        img_Gaus = img_egalisation
        # Init des seuils 
        imgHSV = img_Gaus.copy()# cv2.cvtColor(img_Gaus,cv2.COLOR_BGR2HLS)
    
        print("Check color  : ", color)   
        print("Press any touch when you are satisfied with the current detection")
        
        def binarizeHSV(pos):
            global lower,upper,k1
            Hmin = cv2.getTrackbarPos('Hmin','Mon masque')
            Hmax = cv2.getTrackbarPos('Hmax','Mon masque')
            Smin = cv2.getTrackbarPos('Smin','Mon masque')
            Smax = cv2.getTrackbarPos('Smax','Mon masque')
            Vmin = cv2.getTrackbarPos('Vmin','Mon masque')
            Vmax = cv2.getTrackbarPos('Vmax','Mon masque')
            k1 = cv2.getTrackbarPos('Taille noyaux','Mon masque')
            k = 2*k1 + 1 
            #print(k)
        
           # print(k)
            # lower = np.array([Hmin,Lmin,Smin])
            # upper = np.array([Hmax,Lmax,Smax])
            
            lower = np.array([Hmin,Smin,Vmin])
            upper = np.array([Hmax,Smax,Vmax])
            
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
            T = 0
            L = []
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
            
            imgfinal = cv2.cvtColor(imgfinal,cv2.COLOR_HSV2BGR) 
            
            cv2.circle(imgfinal,(cx,cy), 4, (0,255,100), -1) 
        
            
            
            cv2.imshow("image centroides",imgfinal)
            
        
            cv2.imshow('Mon masque',img_bin)
            
            return(None)

    
    
        
        cv2.namedWindow('Mon masque',cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Hmin','Mon masque',0,180,binarizeHSV)
        cv2.createTrackbar('Hmax','Mon masque',cv2.getTrackbarPos('Hmin','Mon masque'),180,binarizeHSV)
        cv2.createTrackbar('Smin','Mon masque',0,255,binarizeHSV)
        cv2.createTrackbar('Smax','Mon masque',cv2.getTrackbarPos('Smin','Mon masque'),255,binarizeHSV)
        cv2.createTrackbar('Vmin','Mon masque',0,255,binarizeHSV)
        cv2.createTrackbar('Vmax','Mon masque',cv2.getTrackbarPos('Vmin','Mon masque'),255,binarizeHSV)
        cv2.createTrackbar('Taille noyaux','Mon masque',0,18,binarizeHSV)
        # Test
        binarizeHSV(0)
        
        # Creation des barres de défilement
        
        cv2.waitKey(0)                     
        cv2.destroyAllWindows()
  
        Mcolor = np.hstack((lower,upper,k1))
        
        colors_values[color] = Mcolor

    return(colors_values)
    

if __name__ == "__main__":
    fname = 'Data/Rot/image1.png'
    img_C = cv2.imread(fname)
    # check_color(img_C)
    
    colors_values =  check_all_color(img_C, ['R','V'])