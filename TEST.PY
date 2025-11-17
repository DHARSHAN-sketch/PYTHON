import cv2
import numpy as np
import tensorflow 
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from keras.layers import Dropout, Flatten

import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
 
# Load the trained model
model = load_model("model.h5")

# Define preprocessing functions (same as used in training)
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Define your class names
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vehicles'
    elif classNo == 16: return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vehicles over 3.5 metric tons'

# ----------- TESTING -----------
# Change the path to your image
#image_path = r"C:\Users\DHARSHAN S\Downloads\R.png"
#image_path = r"K:\PROJECTS\Dataset\34\34_20366_1577672008.4301453.png"
#image_path = r"K:\PROJECTS\Dataset\39\39_25410_1577672013.0944633.png"
#image_path = r"K:\PROJECTS\Dataset\0\0_9961_1577671998.6192439.png"
image_path = r"C:\Users\DHARSHAN S\Downloads\20speed.webp"

# Load and preprocess
img = cv2.imread(image_path)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)  # add batch and channel dimension

# Predict
predictions = model.predict(img)
classIndex = np.argmax(predictions, axis=1)
predicted_class = getClassName(classIndex[0])

print("Predicted Class:", predicted_class)

# Optional: show image
cv2.imshow("Input Image", cv2.imread(image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()
