#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[22]:


import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,RMSprop
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os


# In[7]:


# argument parser for CL usage
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# In[8]:


# set of labels/classes 
LABELS = set(["footblall", "tennis", "weight_lifting"])

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # if the label of the current image is not part of of the labels
    # are interested in, then ignore the image
    if label not in LABELS:
        continue
    # load the image, convert it to RGB channel ordering, and resize
    # it to be a fixed 224x224 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)


# In[9]:


# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and validation sets using 75% of
# the data for training and the remaining 25% for validation
(trainX, validX, trainY, validY) = train_test_split(data, labels,
test_size=0.25, stratify=labels, random_state=42)


# In[10]:


# add potential augmentation
train_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
valid_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)


# In[19]:


PARAMS = {'lr': 1e-4,
          'dropout': 0.0,
          'batch_size': 32,
          'n_epochs': args['epochs'],
          'optimizer': 'RMSprop',
          'loss': 'categorical_crossentropy',
          'metrics': 'acc',
          'activations': 'relu, softmax',
          'image_input_shape' : (112,112,3)
          }


# In[20]:


model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(112,112,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(PARAMS['dropout']))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[23]:


model.compile(loss=PARAMS['loss'],
              optimizer=RMSprop(lr=PARAMS['lr']),
              metrics=[PARAMS['metrics']])


# In[24]:


model.summary()


# In[ ]:



spe=len(trainX) // PARAMS['batch_size']
val_spe=len(validY) // PARAMS['batch_size']

H = model.fit(
x=train_datagen.flow(trainX, trainY, batch_size=PARAMS['batch_size']),
steps_per_epoch=spe,
validation_data=valid_datagen.flow(validX, validY),
validation_steps=val_spe,
epochs=PARAMS["epochs"])


# In[ ]:


print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=PARAMS['batch_size'])
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# In[ ]:


# serialize the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")
# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

