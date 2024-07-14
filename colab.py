# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lEp63tgPZLfqtSX3siIXPe3toKsbXXt_
"""

!pip install yfinance

!pip install tensorflow

!pip install ta

!pip install matplotlib

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf

from datetime import datetime, timedelta
end = datetime.now()
start = datetime(end.year-5, end.month, end.day)
data = yf.download('GOOD', start, end)

data.head()

# Visualize the data
def visualize_data(data, column):
  plt.figure(figsize=(14, 7))
  plt.title(f'{column} Stock Price History')
  plt.plot(data[column], label=column)
  plt.xlabel('Year')
  plt.ylabel(f'{column} Price USD ($)')
  plt.show()

# Tính các đặc tính bổ sung ROC, RSI, MA
data['ROC'] = data['Close'].pct_change(periods=1)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
data['Moving_Average'] = data['Close'].rolling(window=14).mean()
data = data.dropna()

visualize_data(data, 'Close')

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create training and testing datasets
def create_datasets(data, target_column, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i-look_back:i, :])
        y.append(data[i, target_column])  # predicting the target_column (Close price, ROC, etc.)
    return np.array(x), np.array(y)



scaled_data, scaler = preprocess_data(data)

target_column = 'ROC'
target_column_index = data.columns.get_loc(target_column)

data[target_column]

x, y = create_datasets(scaled_data, target_column_index)



x_train, x_test = x[:int(len(x)*0.8)], x[int(len(x)*0.8):]
y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

# Build and train the LSTM model
def build_and_train_model(x_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

model = build_and_train_model(x_train, y_train)

predictions = model.predict(x_test)

scaler = MinMaxScaler(feature_range=(0, 1))
inv_predictions = scaler.fit_transform(predictions)
inv_predictions = scaler.inverse_transform(predictions)

inv_predictions.shape

scaler = MinMaxScaler(feature_range=(0, 1))
inv_test = scaler.fit_transform(y_test.reshape(-1, 1)) # Reshape y_test to a 2D array
inv_test = scaler.inverse_transform(y_test.reshape(-1, 1))

inv_test.shape

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def calculate_accuracy(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

print(calculate_accuracy(inv_test, inv_predictions))

int(len(x)*0.8)

plt.figure(figsize=(14, 7))
plt.plot(predictions, label='Predicted MA')
plt.plot(y_test, label='Actual MA')
plt.xlabel('Date')
plt.ylabel('MA')
plt.title('MA Prediction')
plt.legend(loc='upper left')
plt.show()

def run_main(target_column):
  target_column_index = data.columns.get_loc(target_column)
  x, y = create_datasets(scaled_data, target_column_index)

  x_train, x_test = x[:int(len(x)*0.8)], x[int(len(x)*0.8):]
  y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

  model = build_and_train_model(x_train, y_train)
  predictions = model.predict(x_test)

  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit_transform(predictions)
  scaler.inverse_transform(predictions)
  scaler.fit_transform(y_test.reshape(-1, 1)) # Reshape y_test to a 2D array
  scaler.inverse_transform(y_test.reshape(-1, 1))

  plt.figure(figsize=(14, 7))
  plt.plot(predictions, label='Predicted MA')
  plt.plot(y_test, label='Actual MA')
  plt.xlabel('Date')
  plt.ylabel('MA')
  plt.title('MA Prediction')
  plt.legend(loc='upper left')
  plt.show()

run_main("ROC")

run_main("Close")
