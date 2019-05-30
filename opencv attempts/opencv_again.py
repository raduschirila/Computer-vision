# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:29:43 2019

@author: radus
"""

import numpy as np
import cv2
img= cv2.imread('profile.jpg',0);
cv2.imshow("initial",img);
imgv=np.array(img,dtype=float)
for i in range(imgv.shape[0]):
    for j in range(imgv.shape[1]):
        if imgv[i][j]>100:
            imgv[i][j]=0;
        else:
            imgv[i][j]=170;
cv2.imshow("s",imgv)
ret, thresh = cv2.threshold(img, 150, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()
