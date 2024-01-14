import os
import keras
import cv2 as cv
from matplotlib import pyplot as plt
# import tensorflow_addons as tfa

import numpy as np
# from keras.models import Model
# from tensorflow import Model
import tf


BASE_PATH = './DataSet'
default_img_shape = 640

def load_data_set(type='train'):
  # Configure paths
  IMG_DIR_PATH = os.path.join(BASE_PATH, type)
  annotations_path = os.path.join(IMG_DIR_PATH, '_annotations.csv')
  # Reading annotations
  annotations_file = open(annotations_path)
  # Convertting annotations to array
  raws = []
  for line in annotations_file: 
    line = line.replace("640,640,Pill,", '')# Delliting usles information
    raws.append(line.replace('\n', '').split(',')) 
  raws.pop(0) # Removing from csv data first~header line 

  out_data = {
    "images": [],
    "targets": []
  }
  for index in range(0, len(raws)): # Converting each number to percentage of image size to cordinates
    temp = list(map(lambda x: int(x)/default_img_shape if x.isnumeric() else x, raws[index]))
    path_to_img = os.path.join(IMG_DIR_PATH, temp[0]) 
    image = cv.imread(path_to_img, cv.IMREAD_GRAYSCALE)
    out_data["images"].append(image)
    out_data["targets"].append(temp[1::])

  return out_data

def init_model(train_img, train_target, test_img, test_target): #predictor):
  model = keras.Sequential()

  # Block 1
  model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", input_shape = (default_img_shape, default_img_shape, 1), activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.MaxPooling2D())
  
  # Block 2
  model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.MaxPooling2D())
  
  
  # Block 3
  model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.MaxPooling2D())

    # Block 4
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.MaxPooling2D())

      # Block 5
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
  model.add(keras.layers.MaxPooling2D())
  
  # Top
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation="relu"))
  model.add(keras.layers.Dense(64, activation="relu"))
  model.add(keras.layers.Dense(32, activation="relu"))
  model.add(keras.layers.Dense(4, activation="sigmoid"))
  
  # model.build()
  # model.summary()

  opt = keras.optimizers.Adam()
  loss_fn = keras.losses.binary_crossentropy
  model.compile(loss=loss_fn, optimizer=opt)
  # print(model.summary())

  H = model.fit(train_img, train_target, validation_data=(test_img, test_target), batch_size=32, epochs=5, verbose=1)


  # result = model.predict(np.array([predictor]))
  # 
  # for i in range(0, 64):
  #   predict_img = result[0, :, :, i]
  #   ax = plt.subplot(8,8,i+1)
  #   ax.set_xticks([])
  #   ax.set_yticks([])
  #   plt.imshow(predict_img, cmap="gray")
  # plt.show()

# def train(model, train_img, train_target):

train_data = load_data_set()
test_data = load_data_set("test")

init_model(train_data["images"], train_data["targets"], test_data["images"], test_data["targets"])