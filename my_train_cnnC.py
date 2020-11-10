from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
np.random.seed(1337)

def read_image(imageName):
    im = Image.open(imageName)
    data = np.array(im)
    return data
#text = os.listdir('F:\\DataSet\\Images_rec\\B')
images=[]
labels=[]
for i in range(1,2001):
    images.append(read_image('/Users/HZK/YBJ/image_recovery/trainC/dog_'+str(i)+'.jpg'))
    labels.append(read_image('/Users/HZK/YBJ/image_recovery/labelC1/dog_'+str(i)+'.jpg'))
X = np.array(images)#.transpose()
y = np.array(labels)#.transpose()#.reshape(-1, 1,28, 28)/255.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
#print (X_train)
# dimensions of our images.
img_width, img_height = 402, 266

# train_data_dir = "F:\\DataSet\\Images_rec\\trainB\\"
# validation_data_dir = "F:\\DataSet\\Images_rec\\validationB\\"
nb_train_samples = 1400
nb_validation_samples = 600
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first': # channels_first means 3(rgb) is in the first element of input_shape
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential() #
model.add(Conv2D(3, (3, 3), strides=1, padding='same', input_shape=input_shape))  # first layer, needs input_shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(12, (3, 3), strides=1, padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3), strides=1, padding='same', input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(UpSampling2D(size=(2, 2), data_format=None))
# model.add(Conv2D(32, (3, 3), strides=1, padding='same', input_shape=input_shape))  # first layer, needs input_shape
# model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2), data_format=None))
model.add(Conv2D(12, (3, 3), strides=1, padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2), data_format=None))
model.add(Conv2D(3, (3, 3), strides=1, padding='same', input_shape=input_shape))
model.add(Activation('relu'))

#fully connected layer
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
model.compile(loss= 'mean_squared_error',#'binary_crossentropy',#'categorical_crossentropy'
              optimizer='rmsprop',#Adam(lr=1e-4)
              metrics=['accuracy'],)
              #target_tensors = "F:\\DataSet\\Images_rec\\train_unB\\")

# this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=None)
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')
#
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

#how to output the image after the convolution neural network?
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,)#1,64

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save_weights("/Users/HZK/YBJ/image_recovery/network/C_4layer_2000.h5")