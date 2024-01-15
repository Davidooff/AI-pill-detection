import os
import keras
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from config import config

class AI:

  def __init__(self, config):
    self.model = keras.Sequential()
    self.dataSet = { # Data set structure
      "train": {
        "images": [],
        "targets": [],
      },
      "test": {
        "images": [],
        "targets": [],
      }
    }
    self.config = config

  def load_data_set(self, dataSet, type="train"):
    self.dataSet[type] = dataSet

  def get_data_set(self, type='train'): # Getting raw data as raws in array
    # Configure paths
    IMG_DIR_PATH = os.path.join(self.config["DATA_SET_PATH"], type)
    annotations_path = os.path.join(IMG_DIR_PATH, '_annotations.csv')
      
    # Convertting annotations to array
    with open(annotations_path) as annotations_file: # Reading annotations
      lines = annotations_file.read()
      lines = lines.split('\n') # spliting every line
    lines.pop(0) # Removing from csv data first~header line 
    return lines

  
  def convert_data_set(self, annotations_data, del_substr="640,640,Pill,", type='train'):
    dataSet = {
      "images": [],
      "targets": []
    }

    for line in annotations_data: 
      line = line.replace(del_substr, '') # Delliting usles information
      line = line.split(',') # [image-name, xmin, ymin, xmax, ymax]
      path_to_img = os.path.join(self.config["DATA_SET_PATH"], type, line[0]) 
      image = cv.imread(path_to_img)
      image = cv.resize(image, (224,224))
      image = image[...,::-1]
      dataSet["images"].append(image / 255.0) 
      converted_targets = list(map(int, line[1::]))
      converted_targets = np.divide(converted_targets, self.config["DEFAULT_IMG_SHAPE"] * 2) #geting line as [xmin, ymin, xmax, ymax] and shape as [x,y] * 2 => [x,y,x,y]
      dataSet["targets"].append(converted_targets)

    dataSet["images"] = np.array(dataSet["images"], dtype="float32")
    dataSet["targets"] = np.array(dataSet["targets"], dtype="float32")
    return dataSet
  
  def init_layers(self):
    # Block 1
    self.model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", input_shape = (224, 224, 3), activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D())
    
    # Block 2
    self.model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D())
    
    
    # Block 3
    self.model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D())

    # Block 4
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D())

    # Block 5
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D())
    
    # Top
    self.model.add(keras.layers.Flatten())
    self.model.add(keras.layers.Dense(256, activation="relu"))
    self.model.add(keras.layers.Dense(128, activation="relu"))
    self.model.add(keras.layers.Dense(64, activation="relu"))
    self.model.add(keras.layers.Dense(32, activation="relu"))
    self.model.add(keras.layers.Dense(4, activation="sigmoid"))
    

    opt = keras.optimizers.Adam(self.config["LEARNING_RATE"])
    loss_fn = keras.losses.binary_crossentropy
    self.model.compile(loss="mse", optimizer=opt)
    print(self.model.summary())


  def fit_data_set(self):
    H = self.model.fit(self.dataSet["train"]["images"], 
                       self.dataSet["train"]["targets"], 
                       validation_data=(self.dataSet["test"]["images"], self.dataSet["test"]["targets"]), 
                       batch_size=self.config["BATCH_SIZE"], 
                       epochs=self.config["EPOCHS"], 
                       verbose=1)
    return H
  
  def show_and_write_result(self, H):
    N = self.config["EPOCHS"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("training.png")
    plt.show()



model = AI(config)
type = "train"
data = model.get_data_set(type)
data = model.convert_data_set(data, type=type)
model.load_data_set(data, type=type)

type = "test"
data = model.get_data_set(type)
data = model.convert_data_set(data, type=type)
model.load_data_set(data, type=type)

model.init_layers()

# print(model.dataSet)



H = model.fit_data_set()
model.show_and_write_result(H)