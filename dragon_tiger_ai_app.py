import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Dragon vs Tiger AI", layout="centered")

# Result Encoder
map_dict = {"D": 0, "T": 1, "DRAW": 2}
reverse_map = {0: "D", 1: "T", 2: "DRAW"}

# Global storage
session = st.session_state

if "model_trained" not in session:
    session.model_trained = False
if "live_results" not in session:
    session.live_results = []
if "lstm" not in session:
    session.lstm = None
if "xgb" not in session:
    session.xgb = None

def prepare_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def train_models(df):
    data = df["Result"].map(map_dict).values
    X, y = prepare_sequences(data, 10)

    # LSTM shape
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, shuffle=False)

    lstm = Sequential([
        LSTM(64, input_shape=(10, 1)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    lstm_acc = lstm.evaluate(X_test, y_test, verbose=0)[1]

    # XGB
    X_xgb = X.reshape(len(X), 10)
    X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(X_xgb, y, test_size=0.2, shuffle=False)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
    xgb.fit(X_train_x, y_train_x)
    y_pred = xgb.predict(X_test_x)
    xgb_acc = accuracy_score(y_test_x, y_pred)

    return lstm, xgb, lstm_acc, xgb_acc

def hybrid_predict(last10):
    X = np.array(last10).reshape(1,10,1)
    lstm_pred = session.lstm.predict(X, verbose=0)[0]
    xgb_pred = session.xgb.predict(np.array(last10).reshape(1,10))[0]

    lstm_choice = np.argmax(lstm_pred)
    lstm_conf = lstm_pred[lstm_choice]

    if lstm_conf < 0.55: 
        return xgb_pred, xgb_pred, 0.55

    return lstm_choice, xgb_pred, lstm_conf

st.header("🐉 Dragon vs 🐯 Tiger — Live AI Predictor")

uploaded = st.file_uploader("Upload CSV (period, Result)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    df["Result"] = df["Result"].astype(str).str.upper().str.strip()

    st.write("✅ CSV Loaded. Click train")

    if st.button("Train Model"):
        session.lstm, session.xgb, lstm_acc, xgb_acc = train_models(df)
        session.model_trained = True
        
        st.success(f"✅ Model Trained")
        st.write(f"📈 LSTM Accuracy: **{lstm_acc:.2f}**")
        st.write(f"📈 XGBoost Accuracy: **{xgb_acc:.2f}**")

if session.model_trained:
    st.subheader("Enter Live Game Results")
    user_input = st.selectbox("Add result", ["", "D", "T", "DRAW"])

    if st.button("Add Result") and user_input:
        session.live_results.append(map_dict[user_input])

    st.write("Recent:", [reverse_map[i] for i in session.live_results[-10:]])

    if len(session.live_results) >= 10:
        pred, backup, conf = hybrid_predict(session.live_results[-10:])
        st.info(f"🧠 AI Predicts Next: **{reverse_map[pred]}** (confidence: {conf:.2f})")

        if conf < 0.60:
            st.warning("⚠️ Low confidence — continue entering results")

