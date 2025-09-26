import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense,MaxPooling1D,Softmax, LSTM, Dropout
from tensorflow.keras import utils
import numpy as np
import pickle
import glob
import pandas as pd
import time

seq_len = 100
vars = 21

model = Sequential()
model.add(LSTM(256, input_shape = (seq_len, vars), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(4, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.load_weights("weights-improvement-01-0.4904-biggeer.keras")


pickle_in = open('C:\\Users\\Administrator\\Desktop\\课程\\P2AI\\movement_classification\\pickle_data\\02_01_worldpos.pickle',"rb")
data = pickle.load(pickle_in)
X = []

for i in range(0,len(data)-seq_len,1):
    X.append(data[i:i+seq_len])


X= np.asarray(X)
X = np.reshape(X, (X.shape[0], X.shape[1],X.shape[2]*X.shape[3]))
X= X/np.max(X)

result = model(X)
result = np.mean(result,axis=0)
print(f"Walk: {result[0]*100}%")
print(f"Run: {result[1]*100}%")
print(f"Jump: {result[2]*100}%")
print(f"Dance: {result[3]*100}%")


