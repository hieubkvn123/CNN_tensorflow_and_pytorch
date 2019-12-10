import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from bokeh.plotting import figure, show
from scipy.special import softmax

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(12)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(24)

        self.fc = nn.Linear(in_features=16*16*24, out_features=num_classes)

        # self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.norm1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.norm2(output)


        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.norm3(output)

        output = self.conv4(output)
        output = self.relu4(output)
        output = self.norm4(output)


        output = output.view(-1, 16*16*24)

        output = self.fc(output)
        # output = self.softmax(output)

        return output

def detect_face(img):
    haar_file = '/Users/yenmm/Desktop/Hieu/Python/opencv/haarcascade_frontalface_default.xml'
    haar = cv2.CascadeClassifier(haar_file)

    faces = haar.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
    if(len(faces) == 0):
        return None, None
    else:
        (x,y,w,h) = faces[0]
        face = img[y:y+h, x:x+w]
        return (x,y,w,h), face

faces = list()
labels = list()

def prepare_training_data():
    data_file = '/Users/yenmm/Desktop/Hieu/Python/data/faces.csv'
    data = pd.read_csv(data_file,header=0)

    paths = data['path']
    labels_in_file = data['label']

    for file in paths:
        im = cv2.imread(file)

        rect, face = detect_face(im)
        face = cv2.resize(face,(32,32))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        faces.append(face)
    for label in labels_in_file:
        labels.append(label)

prepare_training_data()
le = LabelEncoder()
labels = le.fit_transform(labels)

# pair your image and label together
train_data = []
for i in range(len(labels)):
    train_data.append([faces[i], labels[i]])

# preprocess the image
# 1. convert your images to numpy array
# 2. convert your numpy array of images to torch.Tensor
# 3. reshape your array into : len(array) x n_channels x width x height
X = np.array([i[0] for i in train_data])
X = torch.Tensor(X)
X = X.reshape(-1,3,32,32)  # equivalent to reshape(len(train_data), 3, 32, 32)
print(X.shape)

# Convert labels list to Tensor
Y = np.array([i[1] for i in train_data])
target = torch.Tensor(Y) # shape = [24]
target = target.type(torch.LongTensor)

# Creating the model, the optimizer and the loss function
model = SimpleNet(num_classes = 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Run through the epochs
epoch = 0
running_loss = 0.0
running_losses = []
# while this is the first epoch and running loss is not satisfactory
print("training phase".upper())
while epoch < 1 or running_loss > 0.0001:
    epoch += 1
    running_loss = 0.0
    outputs = model(X)

    loss = criterion(outputs, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    print("Epoch : " + str(epoch) + "|Running Loss : " + str(running_loss))
    running_losses.append(running_loss)

print("model is satisfactory".upper())
print("------------------------------------------------------------------")
# Testing the model
test_img = cv2.imread('/Users/yenmm/Desktop/jack.jpg')
rect, face = detect_face(test_img)

face = cv2.resize(face, (32,32))
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
X = np.array([face])
X = torch.Tensor(X)
X = X.reshape(-1,3,32,32)

outputs = model(X)
outputs = outputs.detach().numpy()
# Calculating softmax manually
outputs = softmax(outputs)

print("testing phase".upper())
print("Predicted label : " + str(le.classes_[np.argmax(outputs[0])]))
print("Full output : " + str(outputs))
print("Certainty = {0:.000%}".format(outputs[0][np.argmax(outputs[0])]))

plot = figure(title='CNN loss using PyTorch',x_axis_label = 'Epochs',y_axis_label = 'Running Losses')
plot.line(range(epoch), running_losses, line_width=2)

# Saving the model
torch.save(model, 'model.h5')
show(plot)
