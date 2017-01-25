
# coding: utf-8

# In[2]:

import numpy as np


# In[3]:

np.random.seed(123)


# In[4]:

from keras.models import Sequential


# In[6]:

from keras.layers import Dense, Dropout, Activation, Flatten


# In[7]:

from keras.layers import Convolution2D, MaxPooling2D


# In[8]:

from keras.utils import np_utils


# In[9]:

from keras.datasets import mnist


# In[10]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[11]:

print(X_train.shape)


# In[12]:

#get_ipython().magic('matplotlib inline')
#from matplotlib import pyplot as plt
#plt.imshow(X_train[0])


# In[13]:

X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)


# In[14]:

print(X_train.shape)


# In[15]:

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[16]:

X_train /= 255.0
X_test /= 255.0


# In[18]:

print(y_train.shape)


# In[19]:

print(y_train[:10])


# In[20]:

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# In[21]:

print(Y_train.shape)


# In[34]:

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), dim_ordering='th'))

model.add(Convolution2D(32,3,3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


# In[35]:

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)


# In[ ]:
score = model.evaluate(X_test, Y_test, verbose=0)


