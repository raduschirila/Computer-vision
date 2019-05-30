# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:01:24 2019

BLURRING EXERCISES


@author: RADU
"""

import cv2
import numpy as np
img=cv2.imread('imag.jpg');
img2=cv2.imread('imag.jpg',0);
cv2.imshow('greyscale',img2);
cv2.imshow('original image',img)
blur=cv2.blur(img, (5,5));
blurg=cv2.blur(img2, (5,5));
cv2.imshow('blurred greyscale',blurg);
cv2.imshow('blurred 5x5',blur);
blur2=cv2.blur(img,(50,50));
cv2.imshow('blurred 50x50',blur2)
sharp=cv2.bilateralFilter(blur,90,100,100)
cv2.imshow('sharpened',sharp)

sharp2=cv2.bilateralFilter(blurg,50,80,150);
cv2.imshow('sharp from blurred greyscale',sharp2)
cv2.waitKey(0);
cv2.destroyAllWindows();