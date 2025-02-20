import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch stock data (Example: Apple - AAPL)
stock_symbol = 'AAPL'
stock_data = yf.download(stock_symbol, start='2020-01-01', end='2024-01-01')

# Display first few rows
print(stock_data.head())

# Calculate Moving Averages
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

# Calculate Daily Returns
stock_data['Daily_Return'] = stock_data['Close'].pct_change()

# Calculate Volatility (Rolling Standard Deviation)
stock_data['Volatility_50'] = stock_data['Daily_Return'].rolling(window=50).std()

# Calculate Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI_14'] = compute_rsi(stock_data)

# Prepare Data for LSTM
stock_data['Target'] = stock_data['Close'].shift(-1)  # Predicting next day's closing price
features = ['SMA_50', 'SMA_200', 'EMA_50', 'Daily_Return', 'Volatility_50', 'RSI_14']
stock_data = stock_data[features + ['Target']]
stock_data.dropna(inplace=True)

# Scale the Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data)

# Convert Data into Sequences (LSTMs need sequential input)
def create_sequences(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, :-1])  # Features
        y.append(data[i+time_steps, -1])  # Target
    return np.array(X), np.array(y)

# Define Time Step for Sequence Length
time_steps = 100
X, y = create_sequences(scaled_data, time_steps)

# Split Data into Training and Testing Sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_steps, X.shape[2])),
    LSTM(100, return_sequences=False),
    Dense(50, activation='relu'),
    Dense(1)
])


# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict on Test Data
y_pred = model.predict(X_test)

# Inverse Transform Only the Target Column
y_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), scaled_data.shape[1]-1)), y_pred), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), scaled_data.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Evaluate Performance
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'LSTM RMSE: {rmse_lstm}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(stock_data.index[train_size + time_steps:], y_test, label='Actual Prices', color='blue')
plt.plot(stock_data.index[train_size + time_steps:], y_pred, label='LSTM Predictions', color='red', linestyle='dashed')
plt.title(f'{stock_symbol} Stock Price Prediction - LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
