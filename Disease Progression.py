import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample data generation (for illustration purposes)
np.random.seed(42)
# Simulate 100 patients with 30 timesteps of health metrics
data = np.random.normal(0, 1, (100, 30, 1))

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = np.array([scaler.fit_transform(patient) for patient in data])

# Train-test split
train_size = int(data_scaled.shape[0] * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(train_data.shape[2]))

model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_data, epochs=10, batch_size=32, validation_data=(test_data, test_data), verbose=2)

# Predict future health metrics
predicted = model.predict(test_data)

# Inverse transform the predictions
predicted_inversed = np.array([scaler.inverse_transform(patient) for patient in predicted])

print("Disease progression predictions for test patients:\n", predicted_inversed)
