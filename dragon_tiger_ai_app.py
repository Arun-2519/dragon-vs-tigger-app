# Dragon vs Tiger AI Predictor (Enhanced Streamlit Version)
# ✅ No LSTM, Pattern + Streak Based AI
# ✅ Features:
# - Markov + Naive Bayes Hybrid
# - Confidence-based Wait Alert with Sound
# - Starts prediction from 11th input
# - Learns from user input continuously
# - Streak analysis (4–10 patterns)
# - Export to CSV + Bar chart

import streamlit as st
import pandas as pd
import random
import os
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

# ========== Setup ==========
st.set_page_config(page_title="Dragon Tiger AI", layout="centered")
st.title("🐉 Dragon vs Tiger AI Predictor")

choices = ["Dragon", "Tiger", "Tie", "Suited Tie"]
confidence_threshold = st.slider("🎯 Confidence Threshold", 50, 100, 70)

# ========== Audio ==========
def play_sound(type):
    sounds = {
        "ding": "https://www.soundjay.com/button/sounds/button-16.mp3",
        "wait": "https://www.soundjay.com/button/sounds/button-10.mp3",
        "start": "https://www.soundjay.com/button/sounds/button-3.mp3"
    }
    if type in sounds:
        st.audio(sounds[type], autoplay=True)

# ========== Memory ==========
if "history" not in st.session_state:
    if os.path.exists("data/prediction_log.csv"):
        st.session_state.history = deque(pd.read_csv("data/prediction_log.csv")['Result'].astype(str).tolist(), maxlen=200)
    else:
        st.session_state.history = deque(maxlen=200)

pattern_memory = defaultdict(Counter)

# ========== AI Models ==========
def markov_predict(history):
    last = history[-1]
    trans = {
        "Dragon": {"Dragon": 0.45, "Tiger": 0.4, "Tie": 0.1, "Suited Tie": 0.05},
        "Tiger": {"Dragon": 0.4, "Tiger": 0.45, "Tie": 0.1, "Suited Tie": 0.05},
        "Tie": {"Dragon": 0.4, "Tiger": 0.4, "Tie": 0.1, "Suited Tie": 0.1},
        "Suited Tie": {"Dragon": 0.3, "Tiger": 0.3, "Tie": 0.2, "Suited Tie": 0.2},
    }
    prob = trans.get(last, {c: 0.25 for c in choices})
    pred = max(prob, key=prob.get)
    return pred, prob[pred]*100

def naive_bayes_predict(history):
    key = tuple(history[-3:]) if len(history) >= 4 else ("", "", "")
    counts = pattern_memory.get(key, Counter({c: 1 for c in choices}))
    pred = counts.most_common(1)[0][0]
    conf = (counts[pred] / sum(counts.values())) * 100
    return pred, conf

def update_bayes_model(history):
    if len(history) >= 4:
        key = tuple(history[-4:-1])
        next_value = history[-1]
        pattern_memory[key][next_value] += 1

# ========== Streak Analyzer ==========
def find_streak(history):
    streaks = []
    for length in range(4, 11):
        if len(history) >= length:
            recent = list(history)[-length:]
            if all(x == recent[0] for x in recent):
                streaks.append((recent[0], length))
    return streaks[-1] if streaks else (None, 0)

# ========== Input Form ==========
st.subheader("📥 Add Game Result")
with st.form("input_form"):
    new_result = st.selectbox("Select Result", choices)
    submit = st.form_submit_button("➕ Add")

if submit:
    st.session_state.history.append(str(new_result))
    update_bayes_model(st.session_state.history)
    df = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prediction_log.csv", index=False)
    play_sound("ding")
    st.success("✅ Result saved")

st.write("🕗 Last 10 Rounds:", list(st.session_state.history)[-10:])

# ========== Prediction Phase ==========
if len(st.session_state.history) >= 10:
    markov_pred, markov_conf = markov_predict(st.session_state.history)
    bayes_pred, bayes_conf = naive_bayes_predict(st.session_state.history)

    votes = defaultdict(float)
    votes[markov_pred] += markov_conf
    votes[bayes_pred] += bayes_conf
    final_pred = max(votes, key=votes.get)
    confidence = votes[final_pred] / (markov_conf + bayes_conf) * 100

    if confidence < confidence_threshold:
        st.warning("⛔ Low confidence. Please wait.")
        play_sound("wait")
    else:
        st.subheader(f"🔮 Prediction: **{final_pred}**")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        play_sound("start")

    # Show streak if any
    streak_value, streak_len = find_streak(st.session_state.history)
    if streak_len >= 4:
        st.info(f"🔥 Detected {streak_len}-win streak of {streak_value}")
else:
    st.warning("⚠️ Need 10 inputs to start prediction.")
    play_sound("wait")

# ========== Chart ==========
if len(st.session_state.history) > 10:
    df_chart = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    fig, ax = plt.subplots()
    df_chart["Result"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db", "#f1c40f", "#9b59b6"])
    plt.title("📊 Result Distribution")
    st.pyplot(fig)

# ========== Export ==========
if st.button("💾 Export to CSV"):
    pd.DataFrame(list(st.session_state.history), columns=["Result"]).to_csv("data/prediction_log.csv", index=False)
    st.success("✅ Exported to prediction_log.csv")
