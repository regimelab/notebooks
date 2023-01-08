import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import sys 
import time 
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Prepare the data
# ===============================================

# Model / data parameters
num_classes = 4
input_shape = (64, 64, 1)
X = None
y = None 

# load it up 
with open('data/segments_X.pkl', 'rb') as fX:
  X = np.array(pickle.load(fX))

with open('data/segments_y.pkl', 'rb') as fy:
  y = np.array(pickle.load(fy))

X, y = shuffle(X, y, random_state=0)

ratio=0.8
x_train, x_test = train_test_split(X, train_size=ratio, test_size=1-ratio, shuffle=False)
y_train, y_test = train_test_split(y, train_size=ratio, test_size=1-ratio, shuffle=False)
dmax, dmin = X.max(), X.min()
print(dmax)
print(dmin)


fig,ax=plt.subplots(2,2)
i=0
for img in x_train: 

  if y_train[i] == 0: 
    ax[1,1].imshow(img, cmap='gray', vmin=dmin, vmax=dmax)
    ax[1,1].set_title(y_train[i])

  if y_train[i] == 1: 
    ax[0,1].imshow(img, cmap='gray', vmin=dmin, vmax=dmax)
    ax[0,1].set_title(y_train[i])

  if y_train[i] == 2: 
    ax[1,0].imshow(img, cmap='gray', vmin=dmin, vmax=dmax)
    ax[1,0].set_title(y_train[i])

  if y_train[i] == 3: 
    ax[0,0].imshow(img, cmap='gray', vmin=dmin, vmax=dmax)
    ax[0,0].set_title(y_train[i])
  i+=1

  if i >= 340:
    break

plt.show() 

print(y_train)
print(y_test)
time.sleep(10)


# split 
#split_loc = 300
#x_train = X[:split_loc]
#x_test = X[split_loc:]
X_outofsample = x_test #X[split_loc:]

#y_train = y[:split_loc]
#y_test = y[split_loc:]

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
#x_train = x_train.astype("float32") / 255
#x_test = x_test.astype("float32") / 255

# Make sure images have shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print(y_test)
print(x_train)
time.sleep(1)


# ================================================
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(5, 5), activation="relu", strides=(2, 2)),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu", strides=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

batch_size = 128
epochs = 3000
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_preds = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
