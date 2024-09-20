# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:18:11 2024

@author: Anish
"""

import joblib
from skimage.io import imread
from skimage import io
from skimage.transform import resize
#from skimage.feature import hog
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt

def display(y):
    if y=='0':
        print("Non-fire")
    else:
        print("Fire")


rf = joblib.load('RandomForestFire.pkl')
k3nn = joblib.load('KNNFire.pkl')
width = 90

img = io.imread('nonfireme2.jpg')  #Put the name of your image file here, must be in the same folder
img = resize(img, (width, width))
plt.imshow(img)
data = dict()
data['data'] = []   
data['data'].append(img)
x = np.array(data['data'])
x=x/255.0

nsamples, nx, ny, nrgb = x.shape
x2 = x.reshape((nsamples,nx*ny*nrgb))
y = rf.predict(x2) # 71.4% accuracy
print("Prediction by random forest: ")
display(y)

print("Prediction by K-Nearest Neighbour: ")
y = k3nn.predict(x2) #67.3% accuracy
display(y)