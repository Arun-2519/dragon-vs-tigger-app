# lottery7_big_small_ai.py
"""
BIG vs SMALL Predictor (Only One Target)
- Works like Dragon vs Tiger model
- Input = B or S
- Output = next prediction (B/S)
- Ensemble = LSTM + Markov + Naive Bayes
- Confidence system + recommendation
- Accuracy graph + Excel export + sound alerts

Labels:
 0 = SMALL
 1 = BIG
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, random
from collections import deque
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# TensorFlow LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except:
    st.error("Install TensorFlow: pip install tensorflow")
    st.stop()

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB


# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="Big vs Small AI", layout="centered")
st.title("🎯 BIG vs SMALL — AI Predictor")

HISTORY_FILE = "big_small_history.json"
MODEL_FILE = "big_small_lstm.h5"
REPLAY_MAX = 4000

LABEL_MAP = {"S":0, "B":1}
INV = {0:"S", 1:"B"}

# Sounds
SOUND_HIGH = "https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg"
SOUND_LOW = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
SOUND_LOSS = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"


# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Hyperparameters")
WINDOW = st.sidebar.number_input("Sequence Window", 3, 30, 10)
LR = float(st.sidebar.number_input("Learning Rate", 0.00001, 0.01, 0.0003))
EPOCHS = st.sidebar.number_input("Epochs per update", 1, 5, 1)
BATCH = st.sidebar.number_input("Batch Size", 8, 128, 32)

W_LSTM = st.sidebar.slider("Weight: LSTM", 0.0, 1.0, 0.6)
W_MARKOV = st.sidebar.slider("Weight: Markov", 0.0, 1.0, 0.25)
W_BAYES = st.sidebar.slider("Weight: NaiveBayes", 0.0, 1.0, 0.15)

CONF_THRESHOLD = st.sidebar.slider("Bet Confidence Threshold (%)", 0, 100, 70)


# -----------------------------------
# SESSION INITIALIZATION
# -----------------------------------
def init():
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            st.session_state.history = json.load(open(HISTORY_FILE))
        else:
            st.session_state.history = []

    if "replay" not in st.session_state:
        st.session_state.replay = deque(maxlen=REPLAY_MAX)

    if "markov" not in st.session_state:
        st.session_state.markov = np.ones((2,2), dtype=np.float32)  # 2 classes

    if "freq" not in st.session_state:
        st.session_state.freq = np.ones(2, dtype=np.float32)

    if "nb" not in st.session_state:
        st.session_state.nb = MultinomialNB()
        st.session_state.nb_trained = False

    if "loss_streak" not in st.session_state:
        st.session_state.loss_streak = 0

    if "accuracy" not in st.session_state:
        st.session_state.accuracy = []

    if "log" not in st.session_state:
        st.session_state.log = []

    if "model" not in st.session_state:
        if os.path.exists(MODEL_FILE):
            st.session_state.model = tf.keras.models.load_model(MODEL_FILE)
        else:
            st.session_state.model = build_lstm()


def build_lstm():
    inp = Input(shape=(WINDOW, 2))
    x = LSTM(64)(inp)
    x = Dropout(0.1)(x)
    out = Dense(2, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss="sparse_categorical_crossentropy")
    return m


# -----------------------------------
# HELPERS
# -----------------------------------
def encode(v):
    """One-hot encode Small(0)/Big(1)"""
    vec = np.zeros(2, dtype=np.float32)
    vec[v] = 1.0
    return vec


def add_round(label):
    st.session_state.history.append(label)
    json.dump(st.session_state.history, open(HISTORY_FILE,"w"))

    # update markov
    if len(st.session_state.history) >= 2:
        prev = st.session_state.history[-2]
        cur = st.session_state.history[-1]
        st.session_state.markov[prev, cur] += 1

    st.session_state.freq[label] += 1

    # replay memory
    H = st.session_state.history
    if len(H) >= WINDOW+1:
        seq = H[-(WINDOW+1):-1]
        target = H[-1]
        counts = np.ones(2, dtype=int)
        for v in seq:
            counts[v] += 1
        st.session_state.replay.append((counts, target))


def train_nb():
    if len(st.session_state.replay) < 6:
        return
    X = np.stack([x[0] for x in st.session_state.replay])
    y = np.array([x[1] for x in st.session_state.replay])
    if not st.session_state.nb_trained:
        st.session_state.nb.fit(X,y)
        st.session_state.nb_trained = True
    else:
        st.session_state.nb.partial_fit(X,y)


def predict():
    H = st.session_state.history
    L = len(H)

    lstm_probs = np.ones(2)/2
    if L >= WINDOW:
        seq = np.stack([encode(v) for v in H[-WINDOW:]])
        inp = seq.reshape(1,WINDOW,2)
        lstm_probs = st.session_state.model.predict(inp, verbose=0)[0]

    markov_row = st.session_state.markov[H[-1]] if L>0 else np.array([1,1])
    markov_probs = markov_row / markov_row.sum()

    if st.session_state.nb_trained and L>=WINDOW:
        counts = np.ones(2, dtype=int)
        for v in H[-WINDOW:]:
            counts[v] += 1
        bayes_probs = st.session_state.nb.predict_proba(counts.reshape(1,-1))[0]
    else:
        bayes_probs = np.ones(2)/2

    w_sum = W_LSTM + W_MARKOV + W_BAYES
    final = (W_LSTM*lstm_probs + W_MARKOV*markov_probs + W_BAYES*bayes_probs) / w_sum
    final = final / final.sum()

    return final


def train_lstm():
    H = st.session_state.history
    Xw, Y = [], []
    for i in range(max(0, len(H)-WINDOW-100), len(H)-WINDOW):
        seq = H[i:i+WINDOW]
        target = H[i+WINDOW]
        Xw.append(np.stack([encode(v) for v in seq]))
        Y.append(target)

    if len(Xw)==0:
        return

    X = np.stack(Xw)
    Y = np.array(Y)
    st.session_state.model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    st.session_state.model.save(MODEL_FILE)


def record_accuracy(pred, actual):
    correct = (pred==actual)
    st.session_state.log.append({
        "time": datetime.now().isoformat(),
        "pred": INV[pred],
        "act": INV[actual],
        "conf": float(p_conf),
        "correct": int(correct)
    })
    if correct:
        st.session_state.loss_streak=0
    else:
        st.session_state.loss_streak+=1

    wins = sum([l["correct"] for l in st.session_state.log[-100:]])
    acc = (wins/max(1,min(100,len(st.session_state.log))))*100
    st.session_state.accuracy.append(acc)


def plot_accuracy():
    if not st.session_state.accuracy:
        st.info("No accuracy data yet.")
        return
    plt.figure(figsize=(5,2))
    plt.plot(st.session_state.accuracy[-200:])
    plt.ylim(0,100)
    plt.title("Accuracy (%)")
    st.pyplot(plt.gcf())


# -----------------------------------
# MAIN UI
# -----------------------------------
init()

st.subheader("Enter Result (Small or Big)")

size = st.selectbox("Result", ["Small (S)", "Big (B)"])
if st.button("Add Round"):
    val = LABEL_MAP[size[0]]
    add_round(val)
    train_nb()
    train_lstm()
    st.success(f"Added: {size}")


# LIVE PREDICTION
st.markdown("---")
st.subheader("Live Prediction")

if len(st.session_state.history) < WINDOW:
    st.info(f"Need {WINDOW - len(st.session_state.history)} more rounds to start predictions.")
else:
    probs = predict()
    pred = int(np.argmax(probs))
    p_conf = float(np.max(probs)*100)

    st.metric("Prediction", INV[pred])
    st.metric("Confidence", f"{p_conf:.1f}%")

    if p_conf >= CONF_THRESHOLD:
        st.success(f"Recommended Bet → {INV[pred]} ({p_conf:.1f}%)")
        st.audio(SOUND_HIGH)
    else:
        st.warning(f"WAIT — Only {p_conf:.1f}% confidence")
        st.audio(SOUND_LOW)

    if st.session_state.loss_streak >= 3:
        st.error(f"⚠ LOSS STREAK: {st.session_state.loss_streak}")
        st.audio(SOUND_LOSS)


# CONFIRM & TRAIN
st.markdown("---")
st.subheader("Confirm Previous Prediction")

true_result = st.selectbox("Actual Outcome", ["S","B"], key="actual_confirm")
if st.button("Confirm Prediction"):
    actual = LABEL_MAP[true_result]
    record_accuracy(pred, actual)
    st.success("Recorded and accuracy updated.")


# HISTORY EXPORT
st.markdown("---")
st.subheader("History")

hist_df = pd.DataFrame([INV[v] for v in st.session_state.history], columns=["Result"])
st.dataframe(hist_df.tail(200))

buf = BytesIO()
hist_df.to_excel(buf, index=False)
st.download_button("⬇ Download History", buf.getvalue(), "big_small_history.xlsx")


# ACCURACY GRAPH
st.markdown("---")
st.subheader("Accuracy")
plot_accuracy()

