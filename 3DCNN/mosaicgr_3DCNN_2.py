#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[10]:


get_ipython().system('pip install keras-video-generators')


# In[11]:


get_ipython().system('pip -V')


# In[1]:


import tensorflow 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import keras_video
import matplotlib.pyplot as plt


# In[2]:


PARAMS = {'lr': 1e-4,
          'dropout': 0.7,
          'batch_size': 6,
          'n_epochs': 10,
          'optimizer': 'RMSprop',
          'loss': 'categorical_crossentropy',
          'metrics': 'acc',
          'activations': 'relu, softmax',
          'image_input_shape' : (15,213,283,3)
          }


# Use constomized DataGenerator (non-funtional at this moment)
# 
# Download .py code from here:
# 
# https://gist.github.com/Emadeldeen24/736c33ac2af0c00cc48810ad62e1f54a
# 
# and then import, e.g.:
# 
# from tweaked_ImageGenerator_v2 import ImageDataGenerator
# 

# In[3]:



#from tweaked_ImageGenerator_v2 import ImageDataGenerator
#train_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
#validation_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
#test_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)


#train = train_datagen.flow_from_directory(
#        '/home/emmanouela/Documents/mosAIc.gr/trimmed_data/train',
#        target_size=(213,283),color_mode='rgb',
#        batch_size=PARAMS['batch_size'],
#        class_mode='categorical',frames_per_step=15) 

#validation = validation_datagen.flow_from_directory(
#        '/home/emmanouela/Documents/mosAIc.gr/trimmed_data/validation',
#        target_size=(213,283),color_mode='rgb',
#        batch_size=PARAMS['batch_size'],
#        class_mode='categorical',frames_per_step=15)


#test = test_datagen.flow_from_directory(
#        '/content/lambda_classification_data/test',
#        target_size=(325, 325),
#        batch_size=PARAMS['batch_size'],
#        class_mode='categorical', color_mode="grayscale")


# Use "keras-videos-generators".
# Must be installed first
# 
# !pip install keras-video-generators
# 
# handy info and examples for these generators here:
# 
# https://pypi.org/project/keras-video-generators/
# 
# https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
# 
# https://nbviewer.jupyter.org/github/metal3d/keras-video-generators/blob/master/Example%20of%20usage.ipynbf
# 
# https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00

# In[3]:


orig_w=640
orig_h=850
ww=np.int(orig_w/3.0)
hh=np.int(orig_h/3.0)
print(ww,hh)

train = keras_video.VideoFrameGenerator(batch_size=PARAMS['batch_size'],nb_frames=15,target_shape=(ww,hh),glob_pattern='/data/user/rantsiou_e/mosAIc.gr_data/medium_dataset_mixed/{classname}/*.mp4', split_val=0.2,split_test=0.1)
valid = train.get_validation_generator()
test = train.get_test_generator()


# In[4]:


for images_batch, labels_batch in valid:
    print('Image batches have shape:', images_batch.shape)
    print('Labes batches have shape:', labels_batch.shape)
    break


# In[19]:


# pick one image from one batch and plot it to check X and Y dimensions are the ones I want and not reversed.

import matplotlib.pyplot as plt
image=images_batch[0,0,:,:,:]
plt.imshow(image)


# In[24]:


tensorflow.keras.backend.clear_session()


# In[5]:


model = models.Sequential()
model.add(layers.Conv3D(64, (3,3,3), padding='same',activation='relu',
                        input_shape=(15,ww,hh,3)))
model.add(layers.MaxPooling3D((2,2,2),strides=(1,2,2)))
model.add(layers.Conv3D(128, (3, 3,3), padding='same',activation='relu'))
model.add(layers.MaxPooling3D((2,2, 2),strides=(1,2,2)))
model.add(layers.Conv3D(256, (3, 3,3), padding='same',activation='relu'))
model.add(layers.MaxPooling3D((2,2,2),strides=(2,2,2)))
model.add(layers.Conv3D(256, (3, 3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D((2,2,2),strides=(2,2,2)))
model.add(layers.Conv3D(256, (3, 3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D((2,2,2),strides=(2,2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(PARAMS['dropout']))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(3, activation='softmax'))


# In[6]:


model.compile(loss=PARAMS['loss'],
              optimizer=optimizers.RMSprop(lr=PARAMS['lr']),
              metrics=[PARAMS['metrics']])


# In[7]:


model.summary()


# In[8]:


spe = len(train)
val_spe=len(valid)

history = model.fit(
            train,
            steps_per_epoch=spe,
            epochs=PARAMS['n_epochs'],
            validation_data=valid,
            validation_steps=val_spe) 


# In[27]:


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[28]:


plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[10]:


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[11]:


plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:




