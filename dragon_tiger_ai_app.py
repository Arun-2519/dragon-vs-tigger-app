import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import random
from collections import defaultdict
from io import BytesIO

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding
except ModuleNotFoundError:
    st.error("❌ Required module 'tensorflow' not found. Please ensure it is listed in requirements.txt.")
    st.stop()

st.set_page_config(page_title="🐉⚖️🌟 Dragon Tiger AI (LSTM Powered)", layout="centered")
st.title("🐉 Dragon vs 🌟 Tiger Predictor (World-Class AI)")

# --- Styles ---
st.markdown("""
    <style>
        body { background-color: #0f1117; color: #ffffff; }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Database ---
conn = sqlite3.connect("dragon_tiger.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS game_data (username TEXT, inputs TEXT, prediction TEXT, confidence REAL, actual TEXT, correct TEXT)''')
conn.commit()

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "inputs" not in st.session_state:
    st.session_state.inputs = []
if "log" not in st.session_state:
    st.session_state.log = []
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "model" not in st.session_state:
    st.session_state.model = None

# --- Login ---
def login(user, pwd): return pwd == "1234"
if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.authenticated = True
            st.session_state.username = u
            st.success("✅ Logged in")
        else:
            st.error("❌ Invalid login")
    st.stop()
if st.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Encode/Decode ---
label_map = {'D': 0, 'T': 1, 'TIE': 2}
reverse_map = {v: k for k, v in label_map.items()}

def encode(seq):
    return [label_map[s] for s in seq if s in label_map]

def decode(v):
    return reverse_map.get(v, "")

# --- Build or Load LSTM ---
def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=3, output_dim=10, input_length=10),
        LSTM(32),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Predict ---
def lstm_predict(seq):
    if len(seq) < 10:
        return None, 0
    X = []
    y = []
    for i in range(10, len(seq)):
        X.append(encode(seq[i-10:i]))
        y.append(encode([seq[i]])[0])
    if not X:
        return None, 0
    model = build_lstm_model()
    model.fit(np.array(X), np.array(y), epochs=10, verbose=0)
    X_input = np.array([encode(seq[-10:])])
    pred = model.predict(X_input, verbose=0)
    result = np.argmax(pred)
    return decode(result), round(np.max(pred) * 100)

# --- Learn & Save ---
def save_result(pred, conf, actual):
    correct = "✅" if pred == actual else "❌"
    st.session_state.log.append({
        "Prediction": pred,
        "Confidence": conf,
        "Actual": actual,
        "Correct": correct
    })
    c.execute("INSERT INTO game_data VALUES (?, ?, ?, ?, ?, ?)", (
        st.session_state.username,
        ",".join(st.session_state.inputs),
        pred,
        conf,
        actual,
        correct
    ))
    conn.commit()

# --- Input UI ---
st.subheader("🎮 Add Round Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D", "T", "TIE"])
if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Prediction ---
if len(st.session_state.inputs) >= 10:
    pred, conf = lstm_predict(st.session_state.inputs)
    if pred is None or conf < 85:
        st.warning("⚠️ Low confidence or not enough data. Waiting for pattern...")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        st.subheader("🧠 AI Prediction")
        st.success(f"Prediction: **{pred}** | Confidence: `{conf}%`")

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions. Be cautious!")

        actual = st.selectbox("Enter actual result", ["D", "T", "TIE"])
        if st.button("✅ Confirm & Learn"):
            save_result(pred, conf, actual)
            st.session_state.inputs.append(actual)
            st.session_state.loss_streak = 0 if pred == actual else st.session_state.loss_streak + 1
            st.rerun()
else:
    needed = 10 - len(st.session_state.inputs)
    st.info(f"Enter {needed} more inputs to begin prediction.")

# --- History ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    if st.button("📥 Generate Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(), file_name=f"{st.session_state.username}_history.xlsx")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LSTM + Real-Time Memory")
