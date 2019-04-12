import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger


read the train/test files
train_x = np.genfromtxt('dltraining/datasets/train_x.csv', delimiter=',')
train_y = np.genfromtxt('dl/training/datasets/train_y.csv', delimiter=',')
test_x = np.genfromtxt('dltraining/datasets/test_x.csv', delimiter=',')
test_y = np.genfromtxt('dltraining/datasets/test_y.csv', delimiter=',')

rows, cols = train_x.shape[0],  train_x.shape[1]
test_rows, test_cols = test_x.shape[0], test_x.shape[1]

train_x = train_x.reshape( rows, 1, cols)
train_y = train_y.reshape( rows, 1, 1)
test_x = test_x.reshape( test_rows, 1, test_cols)
test_y = test_y.reshape(test_rows, 1)

# define parameters
epochs = 1000
batch_size, features, timesteps = rows, cols, rows

#if a model checkpoint does not exist
#then load it and continue from there
checkpoint = Path("/dltraining/checkpoint/best_checkpoint.hdf5")
if checkpoint.is_file():
    print("model from checkpoint")
    # file exists
    # define model
    model = Sequential()

    model.add(LSTM(200, activation='relu', input_shape=(1, features)))

    model.add(RepeatVector(1))

    model.add(LSTM(200, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(200, activation='relu')))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mse', optimizer='adam')
    
    #create early stopping callback
    early_stop = EarlyStopping(monitor='loss',
                               mode='min',
                               verbose=1,
                               patience=20)
    
    #create a checkpoint callback
    filepath="/dltraining/checkpoints/"
    checkpt = ModelCheckpoint(filepath+"best_checkpoint.hdf5", 
                              monitor='loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min')

    #call back to log the loss
    csv_logger = CSVLogger(filepath+'log.csv', 
                           append=True, 
                           separator=',')

    callbacks_list = [checkpt, early_stop, csv_logger]

    #load weights
    model.load_weights("/dltraining/checkpoints/best_checkpoint.hdf5")

    # fit network
    model.fit(train_x, train_y,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0,
              callbacks=callbacks_list)
else:
    print("From Main")
    model = Sequential()

    model.add(LSTM(200, activation='relu', input_shape=(1, features)))

    model.add(RepeatVector(1))

    model.add(LSTM(200, activation='relu', return_sequences=True))

    model.add(TimeDistributed( Dense(200, activation='relu')))
    model.add(TimeDistributed( Dense(1, activation='relu')))

    model.compile( loss='mse', optimizer='adam')
    
    #create an early stopping callback
    early_stop = EarlyStopping(monitor='loss',
                               mode='min',
                               verbose=1,
                               patience=20)
    
    #create a checkpoint callback
    filepath="/dltraining/checkpoints/best_checkpoint.hdf5"
    checkpt = ModelCheckpoint(filepath+"best_checkpoint.hdf5", 
                              monitor='loss',
                              verbose=1,
                              save_best_only=True, 
                              mode='min')

    #call back to log the loss
    csv_logger = CSVLogger(filepath+'log.csv', 
                           append=True, 
                           separator=',')
    callbacks_list = [checkpt, early_stop, csv_logger]

    model.fit(train_x, train_y,
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=0, 
              callbacks=callbacks_list)

pred_y = model.predict(test_x)

# plt.figure()
# plt.plot(test_y.reshape(test_y.shape[0], 1))
# plt.plot(pred_y.reshape(pred_y.shape[0], 1))

# plt.savefig('/dltraining/encoder-decoder.jpeg', format="jpeg", dpi=96)

