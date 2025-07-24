import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB
from collections import deque, defaultdict

# --- App Setup ---
st.set_page_config(page_title="🐉 Dragon vs Tiger AI", layout="centered")
st.title("🐉 Dragon vs Tiger Predictor (Lite AI)")

# --- Styling ---
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

# --- Database Setup ---
conn = sqlite3.connect("dragon_tiger.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS game_data (
    username TEXT, inputs TEXT, prediction TEXT,
    confidence REAL, actual TEXT, correct TEXT
)''')
conn.commit()

# --- Session State Defaults ---
for key in ["authenticated", "username", "inputs", "log", "loss_streak"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "authenticated" else "" if key == "username" else [] if key in ["inputs", "log"] else 0

# --- Login ---
def login(user, pwd):
    return pwd == "1234"

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
            st.error("❌ Invalid credentials")
    st.stop()

if st.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Encoding ---
label_map = {'D': 0, 'T': 1, 'TIE': 2}
reverse_map = {v: k for k, v in label_map.items()}

def encode(seq):
    return [label_map[s] for s in seq if s in label_map]

def decode(label):
    return reverse_map.get(label, "")

# --- Prediction Logic ---
def train_model(history):
    if len(history) < 11:
        return None
    X, y = [], []
    for i in range(10, len(history)):
        X.append(encode(history[i-10:i]))
        y.append(label_map[history[i]])
    model = MultinomialNB()
    model.fit(X, y)
    return model

def predict_next(model, recent_seq):
    if len(recent_seq) < 10:
        return None, 0
    X_input = np.array([encode(recent_seq[-10:])])
    pred_proba = model.predict_proba(X_input)[0]
    label = np.argmax(pred_proba)
    confidence = round(pred_proba[label] * 100)
    return decode(label), confidence

# --- Save Result ---
def save_result(pred, conf, actual):
    correct = "✅" if pred == actual else "❌"
    st.session_state.log.append({
        "Prediction": pred, "Confidence": conf,
        "Actual": actual, "Correct": correct
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
st.subheader("🎮 Add Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D", "T", "TIE"])
if st.button("➕ Add"):
    st.session_state.inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Prediction ---
if len(st.session_state.inputs) >= 11:
    model = train_model(st.session_state.inputs)
    pred, conf = predict_next(model, st.session_state.inputs)

    if pred is None or conf < 70:
        st.warning("⚠️ Low confidence or not enough pattern data.")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        st.subheader("🧠 Prediction")
        st.success(f"Prediction: **{pred}** | Confidence: `{conf}%`")

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ 3+ incorrect predictions. Be cautious!")

        actual = st.selectbox("Enter actual result", ["D", "T", "TIE"])
        if st.button("✅ Confirm & Learn"):
            save_result(pred, conf, actual)
            st.session_state.inputs.append(actual)
            st.session_state.loss_streak = 0 if pred == actual else st.session_state.loss_streak + 1
            st.rerun()
else:
    st.info(f"Enter {11 - len(st.session_state.inputs)} more inputs to enable prediction.")

# --- History ---
if st.session_state.log:
    st.subheader("📊 History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    if st.button("📥 Export Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download", data=buf.getvalue(), file_name=f"{st.session_state.username}_history.xlsx")

st.markdown("---")
st.caption("Built with ❤️ | Lite AI Model | No TensorFlow Needed ✅")
