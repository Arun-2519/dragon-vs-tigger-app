# 🐉 Dragon vs Tiger AI Predictor (Without LSTM, Full Version)

import streamlit as st
import pandas as pd
import random
import os
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="🐉 Dragon Tiger AI", layout="centered")
st.title("🐉 Dragon vs Tiger AI Predictor (No LSTM)")

# --- Game Settings ---
choices = ["Dragon", "Tiger", "Tie", "Suited Tie"]
confidence_threshold = st.slider("🎚️ Confidence Threshold", 50, 100, 70)

# --- Load Session State ---
if "history" not in st.session_state:
    if os.path.exists("data/prediction_log.csv"):
        st.session_state.history = deque(pd.read_csv("data/prediction_log.csv")["Result"].tolist(), maxlen=200)
    else:
        st.session_state.history = deque(maxlen=200)

# --- Pattern Memory for Naive Bayes ---
pattern_memory = defaultdict(Counter)

# --- Markov Model ---
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
    return pred, prob[pred] * 100

# --- Naive Bayes Learning ---
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

# --- Input UI ---
st.subheader("📥 Add Game Result")
with st.form("input_form"):
    new_result = st.selectbox("Select Last Result", choices)
    submit = st.form_submit_button("➕ Add")

if submit:
    st.session_state.history.append(new_result)
    update_bayes_model(st.session_state.history)
    df = pd.DataFrame(st.session_state.history, columns=["Result"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prediction_log.csv", index=False)
    st.success("✅ Result Added")

# --- Last Rounds Display ---
st.write("🕗 Last 10 Rounds:", list(st.session_state.history)[-10:])

# --- Prediction Logic ---
if len(st.session_state.history) >= 5:
    markov_pred, markov_conf = markov_predict(st.session_state.history)
    bayes_pred, bayes_conf = naive_bayes_predict(st.session_state.history)

    votes = defaultdict(float)
    votes[markov_pred] += markov_conf
    votes[bayes_pred] += bayes_conf
    final_pred = max(votes, key=votes.get)
    confidence = votes[final_pred] / (markov_conf + bayes_conf) * 100

    st.subheader(f"🔮 Prediction: **{final_pred}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    if confidence < confidence_threshold:
        st.warning("⚠️ Low confidence. Suggest to WAIT.")

# --- Bar Chart ---
if len(st.session_state.history) >= 10:
    df_chart = pd.DataFrame(st.session_state.history, columns=["Result"])
    fig, ax = plt.subplots()
    df_chart["Result"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db", "#f1c40f", "#9b59b6"])
    plt.title("📊 Result Distribution")
    st.pyplot(fig)

# --- Export to CSV ---
if st.button("💾 Export to CSV"):
    df = pd.DataFrame(st.session_state.history, columns=["Result"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prediction_log.csv", index=False)
    st.success("✅ Data saved to prediction_log.csv")
