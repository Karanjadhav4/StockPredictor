import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from plotly import graph_objs as go
import pytz

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="🧠 LSTM Sniper: Visual Candles")

# Standard Lot Sizes
LOT_SIZES = {
    'Nifty 50': 25, 'Bank Nifty': 15, 'Sensex': 10,
    'Reliance': 1, 'HDFC Bank': 1, 'TCS': 1
}

TICKERS = {
    'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK', 'Sensex': '^BSESN',
    'Reliance': 'RELIANCE.NS', 'HDFC Bank': 'HDFCBANK.NS', 'TCS': 'TCS.NS'
}


# --- DATA LOADER ---
@st.cache_data(ttl=60)
def get_data(ticker):
    # Get 5 days of 1-minute data
    data = yf.download(ticker, period='5d', interval='1m')

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)

    # Calculate Volatility (ATR) for Candle Generation later
    data['tr'] = np.maximum((data['High'] - data['Low']),
                            np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                       abs(data['Low'] - data['Close'].shift(1))))
    data['atr'] = data['tr'].rolling(window=14).mean()

    # Fix Timezone (UTC -> IST)
    data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')

    return data


# --- LSTM DATA PREP ---
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# --- MAIN APP ---
st.title("🧠 LSTM Deep Learning: Future Candles")
st.markdown("Generates **Synthetic Future Candlesticks** using Neural Network predictions.")

option = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
symbol = TICKERS[option]
lot_size = LOT_SIZES[option]

with st.spinner('Downloading Market Data...'):
    df = get_data(symbol)

if df is None or df.empty:
    st.error("Market data unavailable.")
    st.stop()

# --- TRAIN LSTM MODEL ---
st.subheader(f"Training Neural Network on {option}...")
progress_bar = st.progress(0)

# 1. Preprocessing
data_close = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# 2. Train/Test Split
train_size = int(len(scaled_data) * 0.95)
train_data = scaled_data[0:train_size, :]

# 3. Create Sequences
LOOK_BACK = 60
X_train, y_train = create_dataset(train_data, LOOK_BACK)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 4. Build Model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(LOOK_BACK, 1)))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train Live (Fast Mode)
model.fit(X_train, y_train, batch_size=64, epochs=3, verbose=0)
progress_bar.progress(100)

# --- RECURSIVE PREDICTION (THE FUTURE) ---
future_preds = []
curr_batch = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)

for i in range(20):  # Predict 20 mins
    pred = model.predict(curr_batch, verbose=0)[0]
    future_preds.append(pred)
    # Update batch with new prediction
    curr_batch = np.append(curr_batch[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# Inverse Scale to get Real Prices
future_prices = scaler.inverse_transform(future_preds).flatten()

# --- GENERATE SYNTHETIC CANDLES ---
# We take the LSTM price curve and "inflate" it into candles using recent volatility (ATR)
last_real_time = df['Datetime'].iloc[-1]
current_atr = df['atr'].iloc[-1]
if np.isnan(current_atr): current_atr = df['Close'].iloc[-1] * 0.001  # Fallback

future_dates = [last_real_time + pd.Timedelta(minutes=i + 1) for i in range(20)]
future_df = pd.DataFrame({'Datetime': future_dates, 'Close': future_prices})

# Logic:
# Open = Previous Close
# Close = LSTM Prediction
# High/Low = Close +/- ATR buffer
future_df['Open'] = future_df['Close'].shift(1)
future_df.loc[0, 'Open'] = df['Close'].iloc[-1]  # Link to real data

future_df['High'] = np.maximum(future_df['Open'], future_df['Close']) + (current_atr * 0.2)
future_df['Low'] = np.minimum(future_df['Open'], future_df['Close']) - (current_atr * 0.2)

# --- PROFIT CALCULATION ---
current_price = df['Close'].iloc[-1]
final_pred_price = future_prices[-1]
diff = final_pred_price - current_price
profit = diff * lot_size
signal = "BUY" if diff > 0 else "SELL"
color = "green" if diff > 0 else "red"

# --- DASHBOARD UI ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"₹{current_price:,.2f}")
c2.metric("LSTM Target (20m)", f"₹{final_pred_price:,.2f}", delta=f"{diff:.2f}")
c3.markdown(f"### Signal: :{color}[{signal}]")
c4.markdown(f"### Profit (1 Lot): :{color}[₹{abs(profit):,.2f}]")

# --- CHART 1: HISTORICAL DATA ---
st.subheader("1. Recent Market Action (Actual)")
fig1 = go.Figure()
zoom_df = df.iloc[-60:]  # Last 1 hour
fig1.add_trace(go.Candlestick(
    x=zoom_df['Datetime'], open=zoom_df['Open'], high=zoom_df['High'],
    low=zoom_df['Low'], close=zoom_df['Close'], name='Actual'
))
fig1.update_layout(height=400, xaxis_title="Time", yaxis_title="Price", template="plotly_dark",
                   xaxis_rangeslider_visible=False)
st.plotly_chart(fig1, use_container_width=True)

# --- CHART 2: FUTURE PREDICTION ---
st.subheader("2. LSTM Predicted Future (Next 20 Mins)")
st.info("These candles are generated by the AI based on the predicted trend and current volatility.")

fig2 = go.Figure()

# Plot the Synthetic Candles
fig2.add_trace(go.Candlestick(
    x=future_df['Datetime'],
    open=future_df['Open'], high=future_df['High'],
    low=future_df['Low'], close=future_df['Close'],
    name='AI Prediction'
))

# Plot the Trend Line through them
fig2.add_trace(go.Scatter(
    x=future_df['Datetime'], y=future_df['Close'],
    mode='lines', line=dict(color='yellow', width=2, dash='dot'),
    name='LSTM Trend Path'
))

fig2.update_layout(height=500, xaxis_title="Future Time", yaxis_title="Predicted Price", template="plotly_dark",
                   xaxis_rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)