#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:58:13 2017

@author: xiaoran
"""
# import the necessary packages
import cv2


def resize(pic_name):    
    ## load the image and show it
    image = cv2.imread(pic_name)
    img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    
    ### we need to keep in mind aspect ratio so the image does not look skewed or distorted -- therefore, we calculate
    ### the ratio of the new image to the old image
    r = 68 / img.shape[1]
    dim = (68, int(img.shape[0] * r))
    # 
    ## perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)
    return resized
    
with open('/home/xiaoran/Dropbox/svm_Over/flipnames.txt') as f:
#with open('test.txt') as f:
    names = f.readlines()
    names = [x.strip() for x in names] 
    
num = 0    
for name in names:
#    arr = ToZero(name)
    resized = resize('/home/xiaoran/Dropbox/svm_Over/process_flip/'+name)
    cv2.imwrite('/home/xiaoran/Dropbox/svm_Over/resized_flip/'+name, resized)