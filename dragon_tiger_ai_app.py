import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    st.error("❌ Required module 'xgboost' not found. Please ensure it is listed in requirements.txt.")
    st.stop()

st.set_page_config(page_title="🐉⚖️🌟 Dragon Tiger AI", layout="centered")
st.title("🐉 Dragon vs 🌟 Tiger Predictor (AI Advanced)")

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

# --- Session State ---
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
    st.session_state.username = ""
    st.rerun()

# --- Encoding Helpers ---
def encode(seq):
    m = {'D': 0, 'T': 1, 'TIE': 2}
    return [m[s] for s in seq if s in m]

def decode(v):
    m = {0: 'D', 1: 'T', 2: 'TIE'}
    return m.get(v, "")

# --- Adaptive Pattern Clustering Prediction ---
def xgb_predict(seq):
    if len(seq) < 10 or len(st.session_state.X_train) < 30:
        return None, 0

    if st.session_state.model is None:
        model = XGBClassifier(eval_metric='mlogloss')
        X = np.array(st.session_state.X_train)
        y = np.array(st.session_state.y_train)
        model.fit(X, y)
        st.session_state.model = model
    else:
        model = st.session_state.model

    X_input = np.array([encode(seq[-10:])])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]
    return decode(pred), round(max(prob) * 100)

# --- Learn Pattern ---
def learn_pattern(seq, actual):
    if len(seq) >= 10:
        st.session_state.X_train.append(encode(seq[-10:]))
        st.session_state.y_train.append(encode([actual])[0])
        st.session_state.model = None  # Invalidate cache
    for l in range(10, 4, -1):
        if len(seq) >= l:
            key = tuple(seq[-l:])
            st.session_state.markov[key][actual] += 1

# --- Input UI ---
st.subheader("🎮 Add Round Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D", "T", "TIE"])
if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Continuous Learning ---
if len(st.session_state.inputs) > 10:
    for i in range(10, len(st.session_state.inputs)):
        history_slice = st.session_state.inputs[i-10:i]
        result = st.session_state.inputs[i]
        encoded_seq = encode(history_slice)
        if len(encoded_seq) == 10:
            st.session_state.X_train.append(encoded_seq)
            st.session_state.y_train.append(encode([result])[0])

# --- Prediction Section ---
if len(st.session_state.inputs) >= 10:
    pred, conf = xgb_predict(st.session_state.inputs)

    st.text(f"Training Balance ➡️ D: {st.session_state.inputs.count('D')} | T: {st.session_state.inputs.count('T')} | TIE: {st.session_state.inputs.count('TIE')}")

    if pred is None or conf < 65:
        st.warning("⚠️ Not enough data or low confidence. Waiting for pattern...")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        st.subheader("🧠 AI Prediction")
        st.success(f"Prediction: **{pred}** | Confidence: `{conf}%`")

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions. Be cautious!")
            st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

        actual = st.selectbox("Enter actual result", ["D", "T", "TIE"])
        if st.button("✅ Confirm & Learn"):
            correct = actual == pred
            st.session_state.log.append({
                "Prediction": pred,
                "Confidence": conf,
                "Actual": actual,
                "Correct": "✅" if correct else "❌"
            })
            learn_pattern(st.session_state.inputs, actual)
            st.session_state.inputs.append(actual)
            st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1
            st.rerun()
else:
    needed = 10 - len(st.session_state.inputs)
    st.info(f"Enter {needed} more inputs to begin prediction." if needed > 0 else "🧠 Learning from data...")

# --- History & Export ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    if st.button("📥 Generate Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(), file_name=f"{st.session_state.username}_history.xlsx")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, XGBoost, Pattern Clustering, and Auto-Learning")
