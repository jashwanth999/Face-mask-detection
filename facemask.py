from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import random
import os
import cv2
import keras
import pickle
import numpy as np
import imutils
import cvlib as cv

DIRECTORY = r"/content/data"
CATEGORIES = ["with_mask","without_mask"]

dataset=[]
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            label = CATEGORIES.index(category)
            arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            new_arr = cv2.resize(arr, (60, 60))
            dataset.append([new_arr, label])
        except Exception as e:
                print(e)
random.shuffle(dataset)
X = []
y = []
for features, label in dataset:
    X.append(features)
    y.append(label)
X = np.array(X)
y = np.array(y)
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
X = X/255
X = X.reshape(-1, 60, 60, 1)
model = Sequential()
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, validation_split=0.1)
model.save("face_mask_detector_2.model", save_format="h5")
CATEGORIES = ["with_mask","without_mask"]
def images(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(1, 60, 60, 1)
    return new_arr
model = keras.models.load_model('/content/face_mask_detector_2.model')
path=r"/content/data/with_mask/with_mask_1006.jpg"
prediction = model.predict([images(path)])
pre=CATEGORIES[prediction.argmax()]
image = cv2.imread(path)
face, confidence = cv.detect_face(image)
f=face[0]
(startX, startY) = f[0], f[1]
(endX, endY) = f[2], f[3] 
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (startX, startY) 
fontScale = 1
color = (255, 0, 0) 
thickness = 2
image = cv2.putText(image, pre, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

img=cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
cv2_imshow(img)