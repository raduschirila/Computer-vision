# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:56:04 2019

SHAPE DETECTION 

@author: radu
"""

import numpy as np
import cv2

img=cv2.imread('shapes.png'); #import grayscale
cv2.imshow('initial image', img);#show it color
grayscale=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
cv2.imshow('grayscale initial', grayscale);
ret,thresh = cv2.threshold(grayscale,127,255,1);
im2,contours,h = cv2.findContours(thresh,1,2);

#contours is a list of all the coordinates of the contours (corners)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx)==5:
        cv2.drawContours(img,[cnt],0,255,3)
    elif len(approx)==3:

        cv2.drawContours(img,[cnt],0,(0,255,0),3)
    elif len(approx)==4:

        cv2.drawContours(img,[cnt],0,(0,0,255),3)

    elif len(approx) > 15:

        cv2.drawContours(img,[cnt],0,(0,255,255),3)

cv2.imshow('processed', img)


#1: for how many polygons there are, find them all and print the total number, and what they are 
#2: draw contour for each over the initial image



cv2.waitKey(0);
cv2.destroyAllWindows();