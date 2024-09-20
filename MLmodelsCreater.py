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
# # helper function - resize image
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
                        im = io.imread(os.path.join(current_path, file), as_gray = True)
                    except Exception as someerror:
                        print("IO error")
                        e+=1
                        continue
                    im2=Image.open(os.path.join(current_path, file)).convert('L')
                    # if im2.mode != 'RGB':
                    #     continue
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)


IMAGE_PATH = 'firedetect'
CLASSES = os.listdir(IMAGE_PATH)
BASE_NAME = 'fire0or1'
WIDTH = 120
 
# load & resize the images
isExist = os.path.exists(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl') 
print(isExist)
if isExist==False:
    resize_all(src=IMAGE_PATH, pklname=BASE_NAME, width=WIDTH, include=CLASSES)

data = joblib.load(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl')
 
IMAGE_PATH = 'Test_Data'
CLASSES = os.listdir(IMAGE_PATH) 
BASE_NAME = 'testingfire'
WIDTH = 120
 
# load & resize the images
isExist = os.path.exists(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl') 
if isExist==False:
    resize_all(src=IMAGE_PATH, pklname=BASE_NAME, width=WIDTH, include=CLASSES)
data2 = joblib.load(f'{BASE_NAME}_{WIDTH}x{WIDTH}px.pkl')
 

print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
 
Counter(data['label'])

# use np.unique to get all unique values in the list of labels
#labels = np.unique(data['label'])
labels = ['0', '1']
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
x = data['data']
#print(data['data'][0][0].shape)
y = data['label']
#a1=0
#b1=0
#c1=0
print(e)



"""x = np.array(data['data']).reshape(-1, WIDTH, WIDTH, 3)
x = x.astype('float32')
x /= 255

x_test = np.array(data2['data']).reshape(-1, WIDTH, WIDTH, 3)
x_test = x_test.astype('float32')
x_test /= 255

y_test = data2['label']
# split - train & validation
#    SIZE = 0.1 
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=SIZE,shuffle=True,random_state= np.random.randint(1,50),)
x_train=x
y_train=y

y_train3 = []
for i in y_train:
    if i == '0':
        y_train3.append(0)
    else:
        y_train3.append(1)

y_test3 = []
for i in y_test:
    if i == '0':
        y_test3.append(0)
    else:
        y_test3.append(1)"""
    # view
#print(f"Training Size: {x_train.shape[0]}\n Validation Size: {x_test.shape[0]}")

    # normalisation
"""x_train = x_train/255.0
x_test = x_test/255.0"""

"""   # reshape the array (4d to 2d)
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples,nx*ny*nrgb))"""

#########CNN###### # WORK IN PROGRESS
from keras.models import Sequential
import tensorflow as tf
#from keras.layers.core import Dense, Flatten, Activation, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, RMSprop, adam
# training = []
# IMG_SIZE = WIDTH
# for category in labels:
#    path = os.path.join("firedetect", category)
#    class_num = labels.index(category)
#    for img in os.listdir(path):
#       try:
#           im = io.imread(os.path.join(path, img))
#       except Exception as someerror:
#           print("IO error")
#           e+=1
#           continue
#       img_array = cv2.imread(os.path.join(path,img))
#       if img_array is not None:
#           new_array = cv2.resize(img_array, (WIDTH, WIDTH))
#           training.append([new_array, class_num])
          
# import random
# random.shuffle(training)

# xl =[]
# yl =[]
# for features, label in training:
#   xl.append(features)
#   yl.append(label)
  
# feature engineering
x_train = pd.read_csv('flattened_imagesx120.csv')
#print(data['data'][0][0].shape)
y_train = data['label']
#X = X.astype('float32')


x_train /= 255
print(x_train.shape)
x_train = x_train.values.reshape(-1, 120, 120, 1)
print(x_train.shape)
from tensorflow.keras.utils import to_categorical
 # convert to one-hot-encoding
y_train = to_categorical(y_train, num_classes = 2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

from sklearn.metrics import confusion_matrix
import itertools
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(5,5), padding='Same', activation = 'relu', input_shape = x_train[0].shape))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

epochs = 20
batch_size = 25

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, validation_data = (x_val, y_val), steps_per_epoch = x_train.shape[0] // batch_size)

plt.plot(history.history['val_loss'], label = 'validation_loss')
plt.title('Test Loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

import seaborn as sns
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# batch_size = 16
# nb_classes = 2
# nb_epochs = 5
# img_rows = WIDTH
# img_cols = WIDTH
# img_channel = 3
# nb_filters = 32
# nb_pool = 2
# nb_conv = 3

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = tf.nn.relu, input_shape = (img_rows, img_cols, 3)),
#     tf.keras.layers.MaxPooling2D((2,2), strides = 2),
#     tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = tf.nn.relu),
#     tf.keras.layers.MaxPooling2D((2,2), strides = 2),
# #    tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation = tf.nn.relu),
#  #   tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(2, activation = tf.nn.softmax)
# ])

# print(model.summary)
# #model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epochs, validation_data = (x_test, y_test), verbose = 1)
# joblib.dump(model, 'CNNFire.pkl')
# score = model.evaluate(x_test, y_test, verbose = 0)
# print("Test Score: ", score[0])
# print("Test accuracy: ", score[1])


"""###### CNN again!
from keras.models import Sequential
import tensorflow as tf
from keras import layers

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))


from keras.utils import to_categorical
y_train2 = to_categorical(y_train, 2)
y_test2 = to_categorical(y_test, 2)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train2, epochs=1, 
                    validation_data=(x_test, y_test2))

test_loss, test_acc = model.evaluate(x_test, y_test2, verbose=2)
print(test_acc)
"""
#####################SVM
"""
from sklearn import svm 
# Defining the parameters grid for GridSearchCV 
param_grid={'C':[1], 
            'gamma':[0.1], 
            'kernel':['rbf']} 
  

svc=svm.SVC(probability=True) 
  """
# Creating a model using GridSearchCV with the parameters grid 
"""
model=GridSearchCV(svc,param_grid)
model.fit(x_train2,y_train)
y_pred = model.predict(x_test2) 
  
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy SVM: {acc}")
joblib.dump(model, 'SVMFire.pkl')
"""
"""from sklearn import svm 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} """
  
"""
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators = 100, criterion="entropy")
model.fit(x_train2,y_train)
y_pred=model.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Random Forest: {acc}")
joblib.dump(model, 'RandomForestFire.pkl')

"""
#import matplotlib.pyplot as plt;
"""
maxx=0
kmaxx=-1
for i in range(701,1000):
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train2,y_train)
    y_pred=knn.predict(x_test2)
    #acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
   # print(i)
    if maxx<accuracy_score(y_test, y_pred):
        maxx=accuracy_score(y_test, y_pred)
        kmaxx=i
print(kmaxx)
print(maxx)     """
# k=3, acc = 67.34% (1-100)
"""from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train2,y_train)
y_pred=knn.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
# print(i)
print(f"Accuracy KNN with k=3: {acc}")
joblib.dump(knn, 'KNNFire.pkl')"""
"""
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train2,y_train)
y_pred=nb.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Naive Bayes classifier: {acc}")
joblib.dump(nb, 'NaiveBayesFire.pkl')

# Creating a support vector classifier 
#svc=svm.SVC(probability=True) 
 """

 
"""from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train2,y_train)
y_pred=dtc.predict(x_test2)
acc = '{:.1%}'.format(accuracy_score(y_test, y_pred))
print(f"Accuracy Naive Decision Tree: {acc}")
"""
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