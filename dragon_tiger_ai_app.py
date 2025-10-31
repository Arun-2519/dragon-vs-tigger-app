import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from io import BytesIO

try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ Required module 'scikit-learn' not found. Please ensure it is listed in requirements.txt.")
    st.stop()

# --- Page Setup ---
st.set_page_config(page_title="🐉⚖️🌟 Dragon Tiger AI", layout="centered")
st.markdown("""
    <style>
    body { background-color: #0f1117; color: #ffffff; }
    .stButton>button { background-color: #9c27b0; color: white; font-weight: bold; }
    .main { background-color: #1c1e26; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🐉 Dragon vs 🌟 Tiger Predictor (AI Powered)")
st.markdown("### 🎯 Predict outcomes based on past patterns using Naive Bayes and Markov logic.")

# --- Session State ---
if "inputs" not in st.session_state: st.session_state.inputs = []
if "X_train" not in st.session_state: st.session_state.X_train = []
if "y_train" not in st.session_state: st.session_state.y_train = []
if "log" not in st.session_state: st.session_state.log = []
if "loss_streak" not in st.session_state: st.session_state.loss_streak = 0
if "markov" not in st.session_state: st.session_state.markov = defaultdict(lambda: defaultdict(int))

# --- Encoding Helpers ---
def encode(seq): return [ {'D': 0, 'T': 1, 'TIE': 2}[s] for s in seq if s in ['D', 'T', 'TIE'] ]
def decode(v): return {0: 'D', 1: 'T', 2: 'TIE'}.get(v, "")

# --- Prediction Logic ---
def predict(seq):
    if len(seq) < 10:
        st.warning("🕐 Need at least 10 rounds to predict.")
        return None, 0
    if len(st.session_state.X_train) < 20:
        st.info(f"📊 Learning... only {len(st.session_state.X_train)} patterns learned. Need 20+.")
        return None, 0
    encoded = encode(seq[-10:])
    clf = MultinomialNB()
    weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
    pred = clf.predict([encoded])[0]
    conf = max(clf.predict_proba([encoded])[0]) * 100
    return decode(pred), round(conf)

# --- Learning ---
def learn(seq, actual):
    if len(seq) >= 10:
        st.session_state.X_train.append(encode(seq[-10:]))
        st.session_state.y_train.append(encode([actual])[0])
    for l in range(10, 4, -1):
        if len(seq) >= l:
            key = tuple(seq[-l:])
            st.session_state.markov[key][actual] += 1

# --- Input UI ---
st.subheader("🎮 Add Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D", "T", "TIE"])
if st.button("Add"):
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
    pred, conf = predict(st.session_state.inputs)
    labels = [decode(y) for y in st.session_state.y_train]
    st.markdown(f"#### Training Balance ➡️ 🐉 D: {labels.count('D')} | 🌟 T: {labels.count('T')} | ⚖️ TIE: {labels.count('TIE')}")

    # Pie Chart for Label Distribution
    fig1, ax1 = plt.subplots()
    ax1.pie([labels.count('D'), labels.count('T'), labels.count('TIE')],
            labels=['D', 'T', 'TIE'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    ax1.set_title("Training Label Distribution")
    st.pyplot(fig1)

    if pred is None or conf < 65:
        st.warning("⚠️ Not enough data or low confidence. Waiting for pattern...")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        st.subheader("🧠 AI Prediction")
        st.success(f"Prediction: **{pred}** | Confidence: {conf}%")

        # Confidence Trend Chart
        if st.session_state.log:
            confs = [entry["Confidence"] for entry in st.session_state.log]
            fig2, ax2 = plt.subplots()
            ax2.plot(confs, marker='o', color='violet')
            ax2.set_title("Confidence Trend")
            ax2.set_ylabel("Confidence %")
            ax2.set_xlabel("Prediction #")
            st.pyplot(fig2)

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions. Be cautious!")

        actual = st.selectbox("Enter actual result", ["D", "T", "TIE"])
        if st.button("Confirm & Learn"):
            correct = actual == pred
            st.session_state.log.append({
                "Prediction": pred,
                "Confidence": conf,
                "Actual": actual,
                "Correct": "✅" if correct else "❌"
            })
            learn(st.session_state.inputs, actual)
            st.session_state.inputs.append(actual)
            st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1
            st.rerun()
else:
    needed = 10 - len(st.session_state.inputs)
    st.info(f"Enter {needed} more inputs to begin prediction." if needed > 0 else "🧠 Learning from data...")

# --- History ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    df["Emoji"] = df["Correct"].map({"✅": "🎯", "❌": "💥"})
    st.dataframe(df)

    if st.button("Download History"):
        buf = BytesIO()
        df.drop(columns=["Emoji"]).to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(), file_name="prediction_history.xlsx")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Naive Bayes, and Pattern Learning")
