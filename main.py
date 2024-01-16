import os
import sys
import keras
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from config import config
# import parseFlags

class AI:

  def __init__(self, config):
    self.model = keras.Sequential()
    self.dataSet = { # Data set structure
      "train": {
        "images": [],
        "targets": [],
        "shapes": []
      },
      "test": {
        "images": [],
        "targets": [],
        "shapes": []
      },
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

  
  def convert_data_set(self, annotations_data, type='train'):
    dataSet = {
      "images": [],
      "targets": [],
      "shapes": []
    }

    for line in annotations_data: 
      # line = line.replace(del_substr, '') # Delliting usles information
      line = line.split(',') # [image-name, xsize, ysize, pill, xmin, ymin, xmax, ymax]
      path_to_img = os.path.join(self.config["DATA_SET_PATH"], type, line[0]) 
      image = cv.imread(path_to_img)
      image = cv.resize(image, (224,224))
      image = image[...,::-1]
      shape = [int(line[1]), int(line[2])]
      dataSet["images"].append(image / 255.0) 
      converted_targets = list(map(int, line[4::]))
      converted_targets = np.divide(converted_targets, shape * 2) #geting line as[xmin, ymin, xmax, ymax]/([x,y]*2=>[x,y,x,y])
      dataSet["targets"].append(converted_targets) # 0 >= converted_targets <= 1 
      dataSet["shapes"].append(shape)

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
    self.model.add(keras.layers.MaxPooling2D((2, 2), 2))

    # Block 4
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D((2, 2), 2))

    # Block 5
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding="same", activation= "relu"))
    self.model.add(keras.layers.MaxPooling2D((2, 2), 2))
    
    # Top
    self.model.add(keras.layers.Flatten())
    # self.model.add(keras.layers.Dense(256, activation="relu"))
    self.model.add(keras.layers.Dense(128, activation="relu"))
    self.model.add(keras.layers.Dense(64, activation="relu"))
    self.model.add(keras.layers.Dense(32, activation="relu"))
    self.model.add(keras.layers.Dense(4, activation="sigmoid"))
    

    opt = keras.optimizers.Adam(self.config["LEARNING_RATE"])
    # loss_fn = keras.losses.binary_crossentropy
    self.model.compile(loss="mse", optimizer=opt)
    print(self.model.summary())


  def fit_data_set(self):
    H = self.model.fit(self.dataSet["train"]["images"], 
                       self.dataSet["train"]["targets"], 
                       validation_data=(self.dataSet["valid"]["images"], self.dataSet["valid"]["targets"]), 
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
  
  def predict(self, x):
    return self.model.predict(x, batch_size=None, verbose="auto", steps=None, callbacks=None)
  
  def draw(self, x, y, shapes, type="valid"):
    for i in range(len(x)):
      path = os.path.join(self.config["DATA_SET_PATH"], type, x[i])
      image = cv.imread(path)
      start = np.array(y[i][:-2:] * shapes[i], dtype="int32")
      end = np.array(y[i][2::] * shapes[i], dtype="int32")
      print(start, end)
      image = cv.rectangle(image, start, end, (255, 0, 0) , 2) 
      cv.imshow("Bim-Bam", image)
      cv.waitKey(0)

  def save_w(self):
    self.model.save_weights("new-tablet.weights.h5")

  def load_w(self):
    self.model.load_weights("tablet.weights.h5")

  def fit_data_in_class(types = ["train", "test"]):
    type = "train"
    data = model.get_data_set(type)
    data = model.convert_data_set(data, type=type)
    model.load_data_set(data, type=type)

    type = "valid"
    data = model.get_data_set(type)
    data = model.convert_data_set(data, type=type)
    model.load_data_set(data, type=type)
  

def start_train():
  H = model.fit_data_set()
  model.save_w()
  model.show_and_write_result(H)

def run_test_on_valid():
  type = 'test'
  data = model.get_data_set(type)
  urls = list(map(lambda x: x.split(',')[0], data))
  data = model.convert_data_set(data, type=type)
  model.load_data_set(data, type=type)
  prediction = model.predict(np.array(model.dataSet[type]["images"]))
  model.draw(urls, prediction, data["shapes"],type)


model = AI(config)
def main(args = []):
  model.fit_data_in_class()
  model.init_layers()
  if "-l" in args:
    model.load_w()

  if ("-t" in args) or ("-t-show" in args) or ("-t-show-save" in args) or ("-t-save" in args):
    H = model.fit_data_set()
    if ("-t-show-save" in args) or ("-t-save" in args):
      model.save_w()
    if ("-t-show-save" in args) or ("-t-show" in args):
      model.show_and_write_result(H)
    
  if "-p" in args:
    run_test_on_valid()






main(sys.argv)
# # model.load_w()
# start_train()
# run_test_on_valid()