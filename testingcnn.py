from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
import sklearn
from keras.models import load_model
import pandas as pd  
from keras.preprocessing import image
from PIL import Image
import os

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# dimensions of our images.
img_width, img_height = 300,300

test_data_dir = './testing_data'
nb_test_samples = 1556
batch_size = 1

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


test_generator.reset()
model=load_model('firstmodel.h5')
print("model loaded")


f1 = open("results.csv",'w')


for root, dirs, files in os.walk("./testing_data", topdown=False):
    if root == "./testing_data":
        for name in dirs:
            TEST_DIR="./testing_data/"+name+"/"  
            img_file=os.listdir(TEST_DIR)
            for f in (img_file):
                img = Image.open(TEST_DIR+f)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                preds = model.predict(x)
                pred_classes = preds.argmax(axis=-1)
                print(name,pred_classes)
                f1.write(name+"\t"+ str(pred_classes) +"\n")

f1.close()
                
