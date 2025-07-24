import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO

# --- Safe import with clear error ---
try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ Required module 'scikit-learn' not found. Please ensure it is listed in requirements.txt.")
    st.stop()

# --- App UI Config ---
st.set_page_config(page_title="🔁🎟️ Dragon Tiger AI", layout="centered")
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

st.title("🔁 Dragon vs 🌟 Tiger Predictor (AI Powered)")

# --- Session State Init ---
def init_state():
    defaults = {
        "authenticated": False,
        "username": "",
        "inputs": [],
        "X_train": [],
        "y_train": [],
        "log": [],
        "loss_streak": 0,
        "markov": defaultdict(lambda: defaultdict(int))
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

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
label_map = {'D': 0, 'T': 1, 'TIE': 2, 'Suited Tie': 3}
reverse_map = {v: k for k, v in label_map.items()}

def encode(seq):
    return [label_map[s] for s in seq if s in label_map]

def decode(v):
    return reverse_map.get(v, "")

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
    counts = {k: seq[-10:].count(k) for k in ['D', 'T', 'TIE', 'Suited Tie']}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_counts[0][1] > sorted_counts[1][1]:
        return sorted_counts[1][0], 60
    return random.choice(['D', 'T']), 55

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
st.subheader("🎮 Add New Result (D / T / TIE / Suited Tie)")
choice = st.selectbox("Latest Game Result", ["D", "T", "TIE", "Suited Tie"])
if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added ➔ {choice}")

# --- Prediction Block ---
if len(st.session_state.inputs) >= 10:
    pred, conf = predict(st.session_state.inputs)

    st.subheader("🧐 AI Prediction")
    st.success(f"Predicted: **{pred}** | Confidence: `{conf}%`")

    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ 3+ wrong predictions in a row. Watch out!")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)
    elif conf >= 85:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
    elif conf <= 60:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)

    actual = st.selectbox("Enter actual result", ["D", "T", "TIE", "Suited Tie"])
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

        st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1

        st.success("📈 Model updated.")
        st.rerun()
else:
    st.warning(f"⏳ Enter {10 - len(st.session_state.inputs)} more to start prediction.")

# --- Log & Export ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    if st.button("📅 Generate Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", buf.getvalue(),
                           file_name=f"{st.session_state.username}_dragon_tiger_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Made with ❤️ | AI + Bayesian + Markov Learning | Now with Suited Tie Support")
