import streamlit as st
import pandas as pd
import random
import os
from collections import deque, defaultdict, Counter

# ========== SOUND ==========
def play_sound(type):
    sounds = {
        "ding": "https://www.soundjay.com/button/sounds/button-16.mp3",
        "wait": "https://www.soundjay.com/button/sounds/button-10.mp3",
        "high_confidence": "https://www.soundjay.com/button/sounds/button-3.mp3"
    }
    if type in sounds:
        st.audio(sounds[type], autoplay=True)

# ========== MARKOV ==========
def markov_predict(history):
    last = history[-1]
    trans = {
        "Dragon": {"Dragon": 0.45, "Tiger": 0.4, "Tie": 0.15},
        "Tiger": {"Dragon": 0.4, "Tiger": 0.5, "Tie": 0.1},
        "Tie": {"Dragon": 0.5, "Tiger": 0.4, "Tie": 0.1}
    }
    prob = trans.get(last, {"Dragon": 0.33, "Tiger": 0.33, "Tie": 0.34})
    pred = max(prob, key=prob.get)
    return pred, prob[pred]*100

# ========== BAYES ==========
pattern_memory = defaultdict(Counter)

def naive_bayes_predict(history):
    key = tuple(history[-3:]) if len(history) >= 4 else ("", "", "")
    counts = pattern_memory.get(key, Counter({"Dragon":1,"Tiger":1,"Tie":1}))
    pred = counts.most_common(1)[0][0]
    total = sum(counts.values())
    conf = (counts[pred] / total) * 100
    return pred, conf

def update_bayes_model(history):
    if len(history) >= 4:
        key = tuple(history[-4:-1])
        next_value = history[-1]
        pattern_memory[key][next_value] += 1

# ========== LSTM PLACEHOLDER ==========
lstm_training_data = deque(maxlen=500)

def lstm_predict(history):
    if len(lstm_training_data) == 0:
        return random.choice(["Dragon", "Tiger", "Tie"]), 55
    return random.choice(["Dragon", "Tiger", "Tie"]), 65

def train_lstm_model(history):
    if len(history) >= 4:
        lstm_training_data.append(tuple(history[-4:]))

# ========== HYBRID ==========
def hybrid_predict(markov, bayes, lstm):
    votes = {}
    for name, conf in [markov, bayes, lstm]:
        votes[name] = votes.get(name, 0) + conf
    final = max(votes, key=votes.get)
    total = sum(votes.values())
    return final, (votes[final] / total) * 100

# ========== MAIN UI ==========
st.set_page_config(page_title="\ud83d\udd81\ufe0f Dragon Tiger AI", layout="centered")
st.title("\ud83d\udd81\ufe0f Dragon vs Tiger AI Predictor")

if "history" not in st.session_state:
    if os.path.exists("data/prediction_log.csv"):
        st.session_state.history = deque(pd.read_csv("data/prediction_log.csv")["Result"].tolist(), maxlen=50)
    else:
        st.session_state.history = deque(maxlen=50)

with st.form("input_form"):
    choice = st.selectbox("Enter Last Result", ["Dragon", "Tiger", "Tie"])
    submit = st.form_submit_button("\u2705 Add Result")

if submit:
    st.session_state.history.append(choice)
    df = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prediction_log.csv", index=False)
    play_sound("ding")

st.write("\ud83d\udcdf Last 10 Rounds:", list(st.session_state.history)[-10:])

if len(st.session_state.history) >= 5:
    markov_pred, markov_conf = markov_predict(st.session_state.history)
    bayes_pred, bayes_conf = naive_bayes_predict(st.session_state.history)
    lstm_pred, lstm_conf = lstm_predict(st.session_state.history)
    final_pred, confidence = hybrid_predict((markov_pred, markov_conf), (bayes_pred, bayes_conf), (lstm_pred, lstm_conf))

    st.subheader(f"\ud83d\udd2e Prediction: **{final_pred}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    if confidence >= 70:
        play_sound("high_confidence")
    elif confidence < 50:
        play_sound("wait")
        st.warning("\u26d4\ufe0f Low confidence. Suggest to WAIT.")

    update_bayes_model(st.session_state.history)
    train_lstm_model(st.session_state.history)

if st.button("\ud83d\udcc5 Export to Excel"):
    df = pd.DataFrame(list(st.session_state.history), columns=["Result"])
    df.to_csv("data/prediction_log.csv", index=False)
    st.success("Saved!")
