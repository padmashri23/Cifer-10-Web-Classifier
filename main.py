import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

(X_train, y_train),(X_val, y_val) = cifar10.load_data()
X_train = X_train / 255
X_val = X_val / 255

y_train = to_categorical(y_train,10)
y_val = to_categorical(y_val,10)

model = Sequential([
    Flatten(input_shape=(32,32,3)),
    Dense(1000,activation='relu'),
    Dense(10,activation='softmax')

])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_val, y_val))
model.save('cifar10_model.h5')
