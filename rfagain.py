# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:30:31 2024

@author: Anish
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import os
import pprint
from collections import Counter
import joblib
from pprint import pprint
import cv2
from PIL import Image

# skimage libraries
from skimage.io import imread
from skimage import io
from skimage.transform import resize
#from skimage.feature import hog
from skimage.transform import rescale

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
e=0;
# helper function - resize image
def resize_all(src, pklname, include, width=150, height=None):
    global e
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})fire0or1 images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(f"Reading images for {subdir} ...")
            current_path = os.path.join(src, subdir)
        
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    try:
                        im = io.imread(os.path.join(current_path, file))
                    except Exception as someerror:
                        print("IO error")
                        e+=1
                        continue
                    im2=Image.open(os.path.join(current_path, file))
                    if im2.mode != 'RGB':
                        continue
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)


IMAGE_PATH = 'firedetect'
CLASSES = os.listdir(IMAGE_PATH)
BASE_NAME = 'fire0or1'
WIDTH = 255
 
# load & resize the images
resize_all(src=IMAGE_PATH, pklname=BASE_NAME, width=WIDTH, include=CLASSES)

data = joblib.load(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl')
 
IMAGE_PATH = 'Test_Data'
CLASSES = os.listdir(IMAGE_PATH) 
BASE_NAME = 'testingfire'
WIDTH = 255
 
# load & resize the images
resize_all(src=IMAGE_PATH, pklname=BASE_NAME, width=WIDTH, include=CLASSES)

data2 = joblib.load(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl')
 

print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
 
Counter(data['label'])

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])
 
# set up the matplotlib figure and axes, based on the number of labels
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15,4)
fig.tight_layout()
 
# make a plot for every label (equipment) type. The index method returns the 
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)
    

# feature engineering
x = np.array(data['data'])
y = np.array(data['label'])
#a1=0
#b1=0
#c1=0
print(e)
x_test=np.array(data2['data'])
y_test = np.array(data2['label'])
# split - train & validation
#    SIZE = 0.1 
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=SIZE,shuffle=True,random_state= np.random.randint(1,50),)
x_train=x
y_train=y
    # view
print(f"Training Size: {x_train.shape[0]}\n Validation Size: {x_test.shape[0]}")

    # normalisation
x_train = x_train/255.0
x_test = x_test/255.0

    # reshape the array (4d to 2d)
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples,nx*ny*nrgb))


#####################SVM
"""from sklearn import svm 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} """
  

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train2,y_train)
y_pred=model.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Random Forest: {acc}")

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train2,y_train)
y_pred=knn.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy KNN with k=7: {acc}")

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train2,y_train)
y_pred=nb.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Naive Bayes classifier: {acc}")
# Creating a support vector classifier 
#svc=svm.SVC(probability=True) 
  
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train2,y_train)
y_pred=dtc.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Naive Decision Tree: {acc}")
# Creating a model using GridSearchCV with the parameters grid  NOT IDEAL
"""
model=GridSearchCV(svc,param_grid)
model.fit(x_train2,y_train)
y_pred = model.predict(x_test2) 
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
#c1+=accuracy_score(y_test, y_pred)
print(f"Accuracy SVM: {acc}")"""
#a1/=10
#b1/=10
#c1/=10
#print("FINAL RESULTS AFTER 10 ITERATIONS: ")
#print(f"Average Accuracy for Random Forrest: {a1}")
#print(f"Average Accuracy for KNN with k=7: {b1}")
#print(f"Average Accuracy for Naive Bayes Classifier: {c1}")