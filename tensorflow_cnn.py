# IN THIS CNN, THERE ARE 3 CHANGES :
# 1. THE IMAGE DIMENSIONS IS DECREASED FROM (100,100) TO (28,28).
# 2. THE NEW CNN WILL HAVE MORE CONV2D LAYERS AND LESS DENSE LAYERS.
# 3. WE USE BINARY THRESHOLD TO PREPROCESS OUR IMAGES.
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from bokeh.plotting import figure, show

def detect_face(img):
    haar_file = '/Users/yenmm/Desktop/Hieu/Python/opencv/haarcascade_frontalface_default.xml'
    haar = cv2.CascadeClassifier(haar_file)

    faces = haar.detectMultiScale(img, scaleFactor = 1.05, minNeighbors = 5)

    if(len(faces) == 0):
        return None, None
    else:
        (x,y,w,h) = faces[0] # Assuming there is only one face per image
        face = img[y:y+h, x:x+w]

        return (x,y,w,h), face

faces = list()
labels = list()

def prepare_training_data():
    data_file = '/Users/yenmm/Desktop/Hieu/Python/data/faces.csv'
    data = pd.read_csv(data_file, header=0).dropna()

    paths = data['path']
    labels_in_file = data['label']
    for path in paths:
        img = cv2.imread(path)
        rect, face = detect_face(img)

        face = cv2.resize(face, (32,32))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        retval, face = cv2.threshold(face, 200, 255, cv2.THRESH_BINARY)

        faces.append(face)

    for label in labels_in_file:
        labels.append(label)

prepare_training_data()

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=(32,32,1)),
        keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu'),
        keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu'),
        keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, kernel_size=(3,3), activation = 'relu'),
        keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation = 'tanh'),
    keras.layers.Dense(4, activation = 'softmax')
])

model.summary()
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = adam,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# This is importanto !!!
# first input channel = 32 x 32 x 1
faces = np.array([faces]).reshape(len(faces),32,32,1)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np.array(labels)

# Training and retrieving the training history
model.fit(faces, labels, epochs = 50)
history = model.history.history

# Testing the model
img_file = '/Users/yenmm/Desktop/Hieu/Python/data/ken/test.jpg'
img = cv2.imread(img_file)
rect, face = detect_face(img)

face = cv2.resize(face, (32,32))
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
retval, face = cv2.threshold(face, 200, 255, cv2.THRESH_TOZERO)

test = np.array([face]).reshape(1,32,32,1)
predictions = model.predict(test)

print(le.classes_[np.argmax(predictions)])

# plotting the running losses of the model
plot = figure(title='CNN loss using Tensorflow',x_axis_label = 'Epochs',y_axis_label = 'Running Losses')
plot.line(range(50), history['loss'], line_width=2)
show(plot)
