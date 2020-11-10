from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy import misc

img_width, img_height = 372, 299##################################
def build_model():
    if K.image_data_format() == 'channels_first': # channels_first means 3(rgb) is in the first element of input_shape
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()  #
    model.add(Conv2D(3, (3, 3), strides=1, padding='same', input_shape=input_shape))  # first layer, needs input_shape
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(12, (3, 3), strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(48, (3, 3), strides=1, padding='same', input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(UpSampling2D(size=(2, 2), data_format=None))
    # model.add(Conv2D(48, (3, 3), strides=1, padding='same', input_shape=input_shape))  # first layer, needs input_shape
    # model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2), data_format=None))
    model.add(Conv2D(12, (3, 3), strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2), data_format=None))
    model.add(Conv2D(3, (3, 3), strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))

    model.compile(loss= 'mean_squared_error',#'binary_crossentropy',#'categorical_crossentropy'
              optimizer='rmsprop',#Adam(lr=1e-4)
              metrics=['accuracy'],)
    return model
model1=build_model()
model1.load_weights('C:/Users/Administrator/Desktop/homework/weight/B.h5')

def read_image(imageName):
    im = Image.open(imageName)
    data = np.array(im)
    return data
images=[]
images.append(read_image("C:/Users/Administrator/Desktop/homework/data/B.png"))
X = np.array(images)
pre = model1.predict(X)

misc.imsave('C:/Users/Administrator/Desktop/homework/resultB/3160104051_B.png', pre[0])