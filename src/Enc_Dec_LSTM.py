import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#read the train/test files
train_x = np.genfromtxt('../data/train_x.csv', delimiter=',')
train_y = np.genfromtxt('../data/train_y.csv', delimiter=',')
test_x = np.genfromtxt('../data/test_x.csv', delimiter=',')
test_y = np.genfromtxt('../data/test_y.csv', delimiter=',')

features, rows, test_rows, batch = train_x.shape[1], train_x.shape[0], 30, 1

train_x = train_x.reshape(rows, batch, features)
train_y = train_y.reshape(rows, 1, 1)
test_x = test_x.reshape(test_rows, 1, features)
test_y = test_y.reshape(test_rows, 1)

# define parameters
verbose, epochs, batch_size = 1, 1, 1 
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')
# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

pred_y = model.predict(test_x)

plt.figure()
plt.plot(test_y.reshape(test_y.shape[0], 1))
plt.plot(pred_y.reshape(pred_y.shape[0], 1))

plt.savefig("encoder-decoder.jpeg", format="jpeg", dpi=96)
