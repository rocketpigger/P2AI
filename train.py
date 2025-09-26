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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use the fourth GPU
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

x  = []
y = []
seq = []
seq_len = 100
    
    
info = pd.read_excel('data_info.xls')
info = info.drop(['SUBJECT from CMU web database'], axis=1)

    
    
walk = info.loc[info['CATEGORY'] == 'walk']
walk = walk['MOTION'].values.tolist()
# walk = walk[:6]

print(walk)

run = info.loc[info['CATEGORY'] == 'run']
run = run['MOTION'].values.tolist()
# run = run[:6]
print(run)


jump = info.loc[info['CATEGORY'] == 'jump']
jump = jump['MOTION'].values.tolist()
# jump = jump[:6]
print(jump)

dance = info.loc[info['CATEGORY'] == 'dance']
dance = dance['MOTION'].values.tolist()
# dance = dance[:6]
print(dance)

tot = walk + run + jump + dance

for ex in tot:
    pickle_in = open('./pickle_data/'+ ex +'_worldpos.pickle',"rb")
    data = pickle.load(pickle_in)

    seq.append(data)
#print(tot)

for fileno in range(len(seq)):
    for i in range(0,len(seq[fileno])-seq_len,1):
        x.append(seq[fileno][i:i+seq_len])
        if(tot[fileno] in walk):
            y.append(0)
        elif(tot[fileno] in run):
            y.append(1)
        elif(tot[fileno] in jump):
            y.append(2)
        else:
            y.append(3)


X= np.asarray(x)
X = np.reshape(X, (X.shape[0], X.shape[1],X.shape[2]*X.shape[3]))
X= X/np.max(X)
print(X.shape)

y = utils.to_categorical(y)

#print(y.shape)
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(4, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-biggeer.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode = 'min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs = 10, batch_size=32, callbacks=callbacks_list)











