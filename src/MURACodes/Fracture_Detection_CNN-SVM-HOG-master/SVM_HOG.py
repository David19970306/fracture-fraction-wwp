#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:34:03 2017

@author: xiaoran
"""
import timeit
start = timeit.default_timer()

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from skimage.feature import hog


imHog_list = []

Cutoff = 100
def ToZero(pic_name): 
    img = cv2.imread(pic_name)
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    img[img > Cutoff] = 0
    return img

with open('/home/xiaoran/Dropbox/svm_Over3/flipnames.txt') as f:
    names = f.readlines()
    names = [x.strip() for x in names] 

num = 0    
for name in names:
    img = ToZero('/home/xiaoran/Dropbox/svm_Over3/resized_flip/'+name)
#    im = cv2.imread('/home/xiaoran/Dropbox/svm/resized/'+name)
#    img = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
#    fd, hog_image = hog(img, orientations= 12, pixels_per_cell=(12, 12),
    fd, hog_image = hog(img, orientations= 10, pixels_per_cell=(25, 25),
                    cells_per_block=(1, 1), visualise=True)
    imHog_list.append(fd)
    num = num + 1
    
label_array = np.zeros((2294,1))
with open('/home/xiaoran/Dropbox/svm_Over3/flipLabel_values.txt') as f:
    vals = f.readlines()
    vals = [x.strip() for x in vals] 
k = 0    
for val in vals:
    label_array[k,0] = val
    k = k+1
   
im_array = np.asarray(imHog_list)

# assign variables to X and y
Xraw = im_array
yraw = label_array

sc = StandardScaler()
sc.fit(Xraw)
Xraw = sc.transform(Xraw)

n = np.arange(500,500)

X = np.delete(Xraw, (n), axis=0)
y = np.delete(yraw, (n), axis=0) 

# split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## principal component analysis for SEM images
n_components = 9
pca = PCA(n_components)

pca.fit(X_train)
pca_train = pca.fit_transform(X_train)
pca_test = pca.fit_transform(X_test)

# calculate and plot explained variance from PCA
def pcaPlot():
    var = pca.explained_variance_ratio_  
    cum_var = np.cumsum(var)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
    
        plt.bar(range(n_components), var, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(n_components), cum_var, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
#pcaPlot()    
## use SVM for image classification on raw data
def svmRaw():
    clf = SVC()
    clf.fit(X_train, y_train.ravel()) 
    y_pred = clf.predict(X_test)
    
    # prediction accuracy
    def predPlot():
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))  
            xx = np.arange(len(y_test))
            plt.scatter(xx,y_test)
            plt.scatter(xx, y_pred)
            plt.ylabel('Predicted Lable')
            plt.xlabel('True Label')
            plt.legend(loc='best')
            plt.tight_layout()
#    predPlot()
    
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    return y_pred
y_pred = svmRaw()


## use SVM for image classification on PCA


def svmPCA():
    clf = SVC()
    clf.fit(pca_train, y_train.ravel()) 
    y_pred = clf.predict(pca_test)
    
    # prediction accuracy
    def predPlot():
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))  
            xx = np.arange(len(y_test))
            plt.scatter(xx,y_test)
            plt.scatter(xx, y_pred)
            plt.ylabel('Predicted Lable')
            plt.xlabel('True Label')
            plt.legend(loc='best')
            plt.tight_layout()
#    predPlot()
#    
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    return y_pred
#y_pred = svmPCA()

#np.savetxt("y_train.csv",y_train, delimiter=',', header="y_train", comments="")

# time the running process
stop = timeit.default_timer()
print (stop - start)