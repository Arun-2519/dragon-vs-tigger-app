import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import deque, defaultdict, Counter
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="🐉 Dragon Tiger AI", layout="centered")

# --- Setup ---
if "history" not in st.session_state:
    if os.path.exists("data/prediction_log.csv"):
        st.session_state.history = deque(pd.read_csv("data/prediction_log.csv")["Result"].tolist(), maxlen=200)
    else:
        st.session_state.history = deque(maxlen=200)

if "X" not in st.session_state: st.session_state.X = []
if "y" not in st.session_state: st.session_state.y = []
if "model" not in st.session_state: st.session_state.model = LogisticRegression()
if "pattern_memory" not in st.session_state: st.session_state.pattern_memory = defaultdict(Counter)
if "streak_alerted" not in st.session_state: st.session_state.streak_alerted = False

# --- Helper ---
def encode(val):
    return {"Dragon": 0, "Tiger": 1, "Tie": 2, "Suited Tie": 3}.get(val, -1)
def decode(val):
    return {0: "Dragon", 1: "Tiger", 2: "Tie", 3: "Suited Tie"}.get(val, "Unknown")

def play_sound(alert_type):
    sounds = {
        "ding": "https://www.soundjay.com/button/sounds/button-16.mp3",
        "wait": "https://www.soundjay.com/button/sounds/button-10.mp3",
        "high": "https://www.soundjay.com/button/sounds/button-3.mp3"
    }
    if alert_type in sounds:
        st.audio(sounds[alert_type], autoplay=True)

# --- Prediction Modules ---
def markov_predict(history):
    last = history[-1]
    trans = {
        "Dragon": {"Dragon": 0.4, "Tiger": 0.4, "Tie": 0.1, "Suited Tie": 0.1},
        "Tiger": {"Tiger": 0.4, "Dragon": 0.4, "Tie": 0.1, "Suited Tie": 0.1},
        "Tie": {"Dragon": 0.35, "Tiger": 0.35, "Tie": 0.2, "Suited Tie": 0.1},
        "Suited Tie": {"Dragon": 0.4, "Tiger": 0.4, "Tie": 0.1, "Suited Tie": 0.1}
    }
    prob = trans.get(last, {"Dragon": 0.25, "Tiger": 0.25, "Tie": 0.25, "Suited Tie": 0.25})
    pred = max(prob, key=prob.get)
    return pred, prob[pred]*100

def naive_bayes_predict(history):
    if len(history) < 4: return random.choice(["Dragon", "Tiger", "Tie", "Suited Tie"]), 50
    key = tuple(history[-3:])
    counts = st.session_state.pattern_memory.get(key, Counter())
    if not counts: return random.choice(["Dragon", "Tiger", "Tie", "Suited Tie"]), 50
    pred, cnt = counts.most_common(1)[0]
    total = sum(counts.values())
    conf = (cnt / total) * 100
    return pred, conf

def update_bayes_model(history):
    if len(history) >= 4:
        key = tuple(history[-4:-1])
        next_val = history[-1]
        st.session_state.pattern_memory[key][next_val] += 1

def train_lstm_model():
    if len(st.session_state.X) >= 10:
        st.session_state.model.fit(st.session_state.X, st.session_state.y)

def lstm_predict(history):
    if len(history) < 14: return random.choice(["Dragon", "Tiger", "Tie", "Suited Tie"]), 50
    input_seq = [encode(val) for val in list(history)[-4:]]
    prob = st.session_state.model.predict_proba([input_seq])[0]
    pred = decode(np.argmax(prob))
    conf = max(prob) * 100
    return pred, conf

def hybrid_vote(preds):
    vote_counter = defaultdict(float)
    for label, conf in preds:
        vote_counter[label] += conf
    final = max(vote_counter, key=vote_counter.get)
    total_conf = vote_counter[final] / (sum(vote_counter.values()) + 1e-5) * 100
    return final, total_conf

# --- Input Form ---
st.title("🐉 Dragon vs Tiger AI Predictor")
threshold = st.slider("🎯 Confidence Threshold", 50, 100, 70)

with st.form("input_form"):
    choice = st.selectbox("📥 Add Game Result", ["Dragon", "Tiger", "Tie", "Suited Tie"])
    submit = st.form_submit_button("✅ Add Result")

if submit:
    st.session_state.history.append(choice)
    update_bayes_model(st.session_state.history)
    if len(st.session_state.history) >= 14:
        encoded = [encode(x) for x in list(st.session_state.history)[-14:-4]]
        target = encode(st.session_state.history[-1])
        st.session_state.X.append(encoded)
        st.session_state.y.append(target)
        train_lstm_model()
    df = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prediction_log.csv", index=False)
    play_sound("ding")

# --- Display History ---
st.subheader("🕗 Last 10 Rounds:")
st.code(list(st.session_state.history)[-10:])

# --- Predict if enough data ---
if len(st.session_state.history) >= 14:
    markov_pred, markov_conf = markov_predict(st.session_state.history)
    bayes_pred, bayes_conf = naive_bayes_predict(st.session_state.history)
    lstm_pred, lstm_conf = lstm_predict(st.session_state.history)

    final_pred, final_conf = hybrid_vote([
        (markov_pred, markov_conf),
        (bayes_pred, bayes_conf),
        (lstm_pred, lstm_conf)
    ])

    st.subheader("🔮 AI Prediction")
    st.markdown(f"**Prediction:** `{final_pred}`")
    st.markdown(f"**Confidence:** `{final_conf:.2f}%`")

    if final_conf < threshold:
        play_sound("wait")
        st.warning("⛔ Low confidence. Suggest to WAIT.")
    elif final_conf >= 85:
        play_sound("high")

    # 🔁 Streak detection
    streak = 1
    last = st.session_state.history[-1]
    for prev in reversed(st.session_state.history[:-1]):
        if prev == last:
            streak += 1
        else:
            break
    if streak >= 4 and not st.session_state.streak_alerted:
        st.warning(f"🔥 `{streak}`-Round Streak Detected: {last}")
        st.session_state.streak_alerted = True
    elif streak < 4:
        st.session_state.streak_alerted = False

# --- Chart ---
if len(st.session_state.history) > 0:
    st.subheader("📊 Result Frequency")
    result_df = pd.DataFrame(st.session_state.history, columns=["Result"])
    freq = result_df["Result"].value_counts()
    fig, ax = plt.subplots()
    freq.plot(kind="bar", color=["green", "red", "blue", "purple"], ax=ax)
    st.pyplot(fig)

# --- Export CSV ---
if st.button("💾 Export CSV"):
    df = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    df.to_csv("data/prediction_log.csv", index=False)
    st.success("📁 Saved as `data/prediction_log.csv`")

st.markdown("---")
st.caption("🤖 Built by Vendra | Hybrid AI Model | Live Pattern Learning | Streamlit Deployment Ready")
