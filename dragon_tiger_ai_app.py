
import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO

# Safe import with clear error for Streamlit Cloud
try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ Required module 'scikit-learn' not found. Please ensure it is listed in requirements.txt.")
    st.stop()

# --- App UI Config ---
st.set_page_config(page_title="🐉🆚🐯 Dragon Tiger AI", layout="centered")
st.markdown("""
    <style>
        body { background-color: #0f1117; color: #ffffff; }
        .stButton>button {
            background-color: #9c27b0;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🐉 Dragon vs 🐯 Tiger Predictor (AI Powered)")

# --- Session State Init ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "inputs" not in st.session_state:
    st.session_state.inputs = []
if "X_train" not in st.session_state:
    st.session_state.X_train = []
if "y_train" not in st.session_state:
    st.session_state.y_train = []
if "log" not in st.session_state:
    st.session_state.log = []
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))

# --- Login ---
def login(u, p): return p == "1234"
if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.authenticated = True
            st.session_state.username = u
            st.success("✅ Login successful")
        else:
            st.error("❌ Incorrect login")
    st.stop()

if st.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Encode/Decode ---
def encode(seq):
    m = {'D': 0, 'T': 1, 'TIE': 2}
    return [m[s] for s in seq if s in m]
def decode(v):
    m = {0: 'D', 1: 'T', 2: 'TIE'}
    return m.get(v, "")

# --- Prediction ---
def predict(seq):
    if len(seq) < 10:
        return fallback(seq)
    encoded = encode(seq[-10:])
    if len(st.session_state.X_train) >= 20:
        clf = MultinomialNB()
        weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
        clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
        pred = clf.predict([encoded])[0]
        conf = max(clf.predict_proba([encoded])[0]) * 100
        return decode(pred), round(conf)
    return fallback(seq)

def fallback(seq):
    d = seq[-10:].count("D")
    t = seq[-10:].count("T")
    tie = seq[-10:].count("TIE")
    if d > t and d > tie: return "T", 60
    elif t > d and t > tie: return "D", 60
    return random.choice(["D", "T"]), 55

# --- Learn from result ---
def learn(seq, actual):
    if len(seq) >= 10:
        st.session_state.X_train.append(encode(seq[-10:]))
        st.session_state.y_train.append(encode([actual])[0])
    for l in range(10, 4, -1):
        if len(seq) >= l:
            key = tuple(seq[-l:])
            st.session_state.markov[key][actual] += 1

# --- Input Interface ---
st.subheader("🎮 Add New Result (D / T / TIE)")
choice = st.selectbox("Latest Game Result", ["D", "T", "TIE"])
if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added ➡️ {choice}")

# --- Prediction Block ---
if len(st.session_state.inputs) >= 10:
    pred, conf = predict(st.session_state.inputs)

    st.subheader("🧠 AI Prediction")
    st.success(f"Predicted: **{pred}** | Confidence: `{conf}%`")

    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ 3+ wrong predictions in a row. Watch out!")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)
    elif conf >= 85:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
    elif conf <= 60:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)

    actual = st.selectbox("Enter actual result", ["D", "T", "TIE"])
    if st.button("✅ Confirm & Learn"):
        correct = actual == pred
        learn(st.session_state.inputs, actual)
        st.session_state.inputs.append(actual)

        st.session_state.log.append({
            "Prediction": pred,
            "Confidence": conf,
            "Actual": actual,
            "Correct": "✅" if correct else "❌"
        })

        if correct:
            st.session_state.loss_streak = 0
        else:
            st.session_state.loss_streak += 1

        st.success("📈 Model updated.")
        st.rerun()
else:
    st.warning(f"⏳ Enter {10 - len(st.session_state.inputs)} more to start prediction.")

# --- Log & Export ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Generate Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", buf.getvalue(),
                           file_name=f"{st.session_state.username}_dragon_tiger_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Made with ❤️ | AI + Bayesian + Markov Learning")
