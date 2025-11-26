# lottery7_single_no_violet_improved.py
"""
Lottery7 Wingo — Single place app (NO VIOLET) — Improved color logic
Rules:
 - Numbers 0-4 => Small (S); 5-9 => Big (B)
 - Even numbers (0,2,4,6,8) => Red (R)
 - Odd numbers (1,3,5,7,9) => Green (G)
Single place only. Ensemble: LSTM + Markov + Naive Bayes for number prediction.
Primary color prediction is derived from the highest-probability number (argmax).
Includes: sound alerts, Excel download, accuracy graph, confidence, betting recommendation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, random
from collections import deque
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# ML
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    st.error("Please install tensorflow: pip install tensorflow")
    st.stop()

try:
    from sklearn.naive_bayes import MultinomialNB
except Exception:
    st.error("Please install scikit-learn: pip install scikit-learn")
    st.stop()

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="Lottery7 (No Violet) — Improved", layout="centered")
st.title("🎯 Lottery7 Wingo — Single Place (No Violet) — Improved")

# Files
HISTORY_FILE = "history_single_no_violet.json"
MODEL_NUM_FILE = "model_num_single_no_violet.h5"
REPLAY_MAX = 5000

# Game constants
NUM_CLASSES_NUM = 10
INV_COLOR = {0: "G", 1: "R"}  # 0 -> Green (odd), 1 -> Red (even)
ODD_SET = {1, 3, 5, 7, 9}
EVEN_SET = {0, 2, 4, 6, 8}

# Sidebar hyperparams
st.sidebar.header("Hyperparameters & Controls")
WINDOW = st.sidebar.number_input("Sliding window length", value=10, min_value=3, max_value=30)
CONF_THRESHOLD = st.sidebar.slider("Recommendation confidence %", 0, 100, 70)
LR = float(st.sidebar.number_input("Learning rate", value=1e-4, format="%.6f"))
BATCH_SIZE = st.sidebar.number_input("Batch size", value=32)
INCREMENTAL_EPOCHS = st.sidebar.number_input("Epochs per update", value=1, min_value=1, max_value=3)

# Ensemble weights
W_LSTM = st.sidebar.slider("Weight: LSTM", 0.0, 1.0, 0.6)
W_MARKOV = st.sidebar.slider("Weight: Markov", 0.0, 1.0, 0.25)
W_BAYES = st.sidebar.slider("Weight: NaiveBayes", 0.0, 1.0, 0.15)
if (W_LSTM + W_MARKOV + W_BAYES) <= 0:
    st.sidebar.error("Ensemble weights must sum > 0. Adjust sliders.")
    st.stop()

# Sounds
SOUND_HIGH = "https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg"
SOUND_LOW = "https://actions.google.com/sounds/v1/alarms/warning.ogg"
SOUND_LOSS = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"

# -------------------------
# Helper functions & session init
# -------------------------
def ensure_session():
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                st.session_state.history = json.load(open(HISTORY_FILE, "r"))
            except Exception:
                st.session_state.history = []
        else:
            st.session_state.history = []
    if "replay" not in st.session_state:
        st.session_state.replay = deque(maxlen=REPLAY_MAX)  # store (counts,target)
    if "markov" not in st.session_state:
        st.session_state.markov = np.ones((NUM_CLASSES_NUM, NUM_CLASSES_NUM), dtype=np.float32)
    if "freq_num" not in st.session_state:
        st.session_state.freq_num = np.ones(NUM_CLASSES_NUM, dtype=np.float32)
    if "model_num" not in st.session_state:
        if os.path.exists(MODEL_NUM_FILE):
            try:
                st.session_state.model_num = tf.keras.models.load_model(MODEL_NUM_FILE)
                tf.keras.backend.set_value(st.session_state.model_num.optimizer.learning_rate, LR)
            except Exception:
                st.session_state.model_num = build_number_model()
        else:
            st.session_state.model_num = build_number_model()
    if "nb" not in st.session_state:
        st.session_state.nb = MultinomialNB()
        st.session_state.nb_trained = False
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "log" not in st.session_state:
        st.session_state.log = []
    if "loss_streak" not in st.session_state:
        st.session_state.loss_streak = 0
    if "accuracy_history" not in st.session_state:
        st.session_state.accuracy_history = []

# -------------------------
# Models
# -------------------------
def build_number_model():
    tf.keras.backend.clear_session()
    FEATURES = NUM_CLASSES_NUM + 1  # number one-hot + size bit
    inp = Input(shape=(WINDOW, FEATURES))
    x = LSTM(128)(inp)
    x = Dropout(0.1)(x)
    out = Dense(NUM_CLASSES_NUM, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss="sparse_categorical_crossentropy")
    return m

# -------------------------
# Encoding helpers
# -------------------------
def encode_round_num(entry):
    """Encode a round for LSTM input: one-hot number (10) + size bit (1)"""
    vec = np.zeros(NUM_CLASSES_NUM + 1, dtype=np.float32)
    n = int(entry["num"])
    vec[n] = 1.0
    vec[-1] = 1.0 if n >= 5 else 0.0  # size bit derived from number
    return vec

def color_from_number_idx(n_idx):
    """Return color index: 0 -> Green (odd), 1 -> Red (even)"""
    return 0 if int(n_idx) % 2 == 1 else 1

# -------------------------
# Replay & NB helpers
# -------------------------
def update_replay_with_latest():
    H = st.session_state.history
    if len(H) >= WINDOW + 1:
        window = H[-(WINDOW+1):-1]
        counts = np.ones(NUM_CLASSES_NUM, dtype=int)  # Laplace smoothing
        for r in window:
            counts[int(r["num"])] += 1
        target = int(H[-1]["num"])
        st.session_state.replay.append((counts, target))
        st.session_state.freq_num[target] += 1
        prev_num = int(H[-2]["num"])
        st.session_state.markov[prev_num, target] += 1

def train_nb_from_replay():
    if len(st.session_state.replay) < 8:
        return
    X = np.stack([item[0] for item in st.session_state.replay], axis=0)
    y = np.array([item[1] for item in st.session_state.replay], dtype=int)
    if st.session_state.nb_trained:
        try:
            st.session_state.nb.partial_fit(X, y)
        except Exception:
            st.session_state.nb.fit(X, y)
            st.session_state.nb_trained = True
    else:
        st.session_state.nb.fit(X, y)
        st.session_state.nb_trained = True

# -------------------------
# Prediction ensemble (IMPROVED)
# -------------------------
def predict_next():
    H = st.session_state.history
    L = len(H)
    num_probs_lstm = np.ones(NUM_CLASSES_NUM) / NUM_CLASSES_NUM

    # LSTM prediction
    if L >= WINDOW:
        inp = np.stack([encode_round_num(r) for r in H[-WINDOW:]], axis=0).reshape(1, WINDOW, -1)
        try:
            num_probs_lstm = st.session_state.model_num.predict(inp, verbose=0)[0]
        except Exception:
            num_probs_lstm = np.ones(NUM_CLASSES_NUM) / NUM_CLASSES_NUM

    # Markov (last -> next)
    if L >= 1:
        last = int(H[-1]["num"])
        row = st.session_state.markov[last].astype(np.float32)
        markov_probs = row / row.sum()
    else:
        markov_probs = np.ones(NUM_CLASSES_NUM) / NUM_CLASSES_NUM

    # Bayes (counts in window)
    if st.session_state.nb_trained and L >= WINDOW:
        window = H[-WINDOW:]
        counts = np.ones(NUM_CLASSES_NUM, dtype=int)
        for r in window:
            counts[int(r["num"])] += 1
        try:
            bayes_probs = st.session_state.nb.predict_proba(counts.reshape(1, -1))[0]
        except Exception:
            bayes_probs = np.ones(NUM_CLASSES_NUM) / NUM_CLASSES_NUM
    else:
        bayes_probs = np.ones(NUM_CLASSES_NUM) / NUM_CLASSES_NUM

    # Ensemble combine with smoothing to avoid ties
    eps = 1e-9
    wsum = (W_LSTM + W_MARKOV + W_BAYES)
    num_probs_raw = (W_LSTM * num_probs_lstm + W_MARKOV * markov_probs + W_BAYES * bayes_probs) / wsum
    num_probs = num_probs_raw + eps
    num_probs = num_probs / num_probs.sum()

    # --- DERIVE COLOR ---
    # Method A (aggregated): sum odd/even mass
    green_prob = sum([num_probs[i] for i in range(NUM_CLASSES_NUM) if i in ODD_SET])
    red_prob = sum([num_probs[i] for i in range(NUM_CLASSES_NUM) if i in EVEN_SET])
    color_probs_agg = np.array([green_prob, red_prob], dtype=np.float32)
    color_probs_agg = color_probs_agg / (color_probs_agg.sum() + eps)

    # Method B (argmax-number): pick most-likely number and deterministically derive color
    argmax_num = int(np.argmax(num_probs))
    color_idx_from_argmax = color_from_number_idx(argmax_num)

    # Color confidence (we provide both)
    # confidence_agg = max fraction for either color (agg)
    confidence_agg = float(np.max(color_probs_agg) * 100.0)
    # confidence_argmax = probability of argmax number expressed as percent
    confidence_argmax = float(np.max(num_probs) * 100.0)

    # Choose primary color recommendation: use argmax-based mapping (more decisive)
    primary_color_idx = color_idx_from_argmax
    primary_color_conf = confidence_argmax

    # Size derived deterministically from number_probs (small vs big)
    small_prob = num_probs[:5].sum()
    big_prob = num_probs[5:].sum()
    size_probs = np.array([small_prob, big_prob], dtype=np.float32)
    size_probs = size_probs / (size_probs.sum() + eps)

    # confidences
    confidences = {
        "Number": float(np.max(num_probs) * 100.0),
        "Color_agg": confidence_agg,
        "Color_from_number": confidence_argmax,
        "Size": float(np.max(size_probs) * 100.0)
    }

    # Also compute top-3 number candidates for display
    top_indices = np.argsort(-num_probs)[:3]
    top3 = [(int(i), float(num_probs[i]*100.0)) for i in top_indices]

    return num_probs, {"agg": color_probs_agg, "from_argmax": primary_color_idx}, size_probs, confidences, top3

# -------------------------
# Accuracy tracking
# -------------------------
def record_confirmation(predictions, actual):
    correct_num = (predictions["num"] == actual["num"])
    correct_col = (predictions["col"] == color_from_number_idx(actual["num"]))
    correct_size = (predictions["size"] == (1 if actual["num"] >= 5 else 0))
    st.session_state.log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "pred_num": int(predictions["num"]),
        "act_num": int(actual["num"]),
        "pred_col": int(predictions["col"]),
        "act_col": int(color_from_number_idx(actual["num"])),
        "pred_size": int(predictions["size"]),
        "act_size": int(1 if actual["num"] >= 5 else 0),
        "num_correct": int(correct_num),
        "col_correct": int(correct_col),
        "size_correct": int(correct_size),
        "conf_num": float(predictions["confidences"]["Number"]),
        "conf_col": float(predictions["confidences"]["Color_from_number"]),
        "conf_size": float(predictions["confidences"]["Size"])
    })
    if correct_num:
        st.session_state.loss_streak = 0
    else:
        st.session_state.loss_streak += 1

    logs = st.session_state.log
    wins = sum([l["num_correct"] for l in logs[-100:]])
    acc = (wins / max(1, min(len(logs), 100))) * 100.0
    st.session_state.accuracy_history.append((datetime.utcnow().isoformat(), acc))

def plot_accuracy_history():
    hist = st.session_state.accuracy_history[-200:]
    if not hist:
        st.info("No accuracy data yet (confirm rounds to build accuracy).")
        return
    times = [pd.to_datetime(t[0]) for t in hist]
    accs = [t[1] for t in hist]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(times, accs)
    ax.set_ylim(0, 100)
    ax.set_title("Rolling Number Accuracy (%)")
    ax.set_ylabel("%")
    ax.set_xlabel("Time")
    st.pyplot(fig)

# -------------------------
# Main UI
# -------------------------
ensure_session()

st.subheader("Enter round (Single place — P1)")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    size_ui = st.selectbox("Size (S/B)", ["S","B"])
with c2:
    color_ui = st.selectbox("Color (for record)", ["G","R"])
with c3:
    num_ui = st.number_input("Number (0-9)", min_value=0, max_value=9, value=0, step=1)

if st.button("Queue & Add round"):
    entry = {"num": int(num_ui), "color": color_ui, "size_observed": 1 if size_ui == "B" else 0}
    st.session_state.history.append(entry)
    json.dump(st.session_state.history, open(HISTORY_FILE, "w"), indent=2)
    update_replay_with_latest()
    train_nb_from_replay()
    st.success("Round added and priors updated.")

# Live prediction & recommendation
st.markdown("---")
st.subheader("Live prediction & recommendation")

num_probs, color_info, size_probs, confidences, top3 = predict_next()
pred_num = int(np.argmax(num_probs))
pred_col_from_argmax = int(color_info["from_argmax"])  # 0: G, 1: R
pred_col_agg = int(np.argmax(color_info["agg"]))
pred_size = int(np.argmax(size_probs))

# Show main metrics
st.metric("Predicted Number", f"{pred_num}", delta=f"conf {confidences['Number']:.1f}%")
st.metric("Predicted Color (from top-number)", f"{INV_COLOR[pred_col_from_argmax]}", delta=f"conf {confidences['Color_from_number']:.1f}%")
st.metric("Predicted Color (aggregated odd/even)", f"{INV_COLOR[pred_col_agg]}", delta=f"conf {confidences['Color_agg']:.1f}%")
st.metric("Predicted Size", "B" if pred_size==1 else "S", delta=f"conf {confidences['Size']:.1f}%")

# Show top-3 numbers to give more context
st.write("Top 3 number candidates (number : prob%)")
for n, p in top3:
    st.write(f"- {n} : {p:.1f}%")

# Recommendation logic
# prefer Number bet when its confidence is highest; prefer Color_from_argmax as next
category_confidences = {
    "Number": confidences["Number"],
    "Color_from_number": confidences["Color_from_number"],
    "Size": confidences["Size"]
}
best_cat = max(category_confidences.keys(), key=lambda k: category_confidences[k])
best_conf = category_confidences[best_cat]
if best_conf >= CONF_THRESHOLD:
    if best_cat == "Number":
        bet_val = str(pred_num)
    elif best_cat == "Color_from_number":
        bet_val = INV_COLOR[pred_col_from_argmax]
    else:
        bet_val = "B" if pred_size==1 else "S"
    st.success(f"RECOMMENDED BET → {best_cat}: {bet_val} (confidence {best_conf:.1f}%)")
    st.audio(SOUND_HIGH)
else:
    st.warning(f"WAIT — highest confidence {best_cat} only {best_conf:.1f}% (<{CONF_THRESHOLD}%). Keep collecting data.")
    st.audio(SOUND_LOW)

if st.session_state.loss_streak >= 3:
    st.error(f"LOSS STREAK: {st.session_state.loss_streak} wrong in a row — be careful.")
    st.audio(SOUND_LOSS)

# Confirm & Learn
st.markdown("---")
st.subheader("Confirm prediction (enter actual result to learn)")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    actual_size = st.selectbox("Actual Size", ["S","B"], key="actual_size")
with c2:
    actual_color = st.selectbox("Actual Color", ["G","R"], key="actual_color")
with c3:
    actual_num = st.number_input("Actual Number (0-9)", min_value=0, max_value=9, value=0, step=1, key="actual_num")

if st.button("Confirm & Learn (label)"):
    actual_entry = {"num": int(actual_num), "color": actual_color, "size_observed": 1 if actual_size=="B" else 0}
    st.session_state.history.append(actual_entry)
    json.dump(st.session_state.history, open(HISTORY_FILE, "w"), indent=2)
    update_replay_with_latest()
    train_nb_from_replay()

    # Build dataset of windows and train LSTM incrementally
    H = st.session_state.history
    windows_X = []
    windows_y = []
    for i in range(0, max(0, len(H) - WINDOW)):
        win = H[i:i+WINDOW]
        Xw = np.stack([encode_round_num(r) for r in win], axis=0)
        y = int(H[i+WINDOW]["num"])
        windows_X.append(Xw)
        windows_y.append(y)
    if len(windows_X) > 0:
        X_arr = np.stack(windows_X, axis=0)
        y_arr = np.array(windows_y, dtype=int)
        try:
            st.session_state.model_num.fit(X_arr, y_arr, epochs=INCREMENTAL_EPOCHS, batch_size=max(8, BATCH_SIZE), verbose=0)
            st.session_state.model_num.save(MODEL_NUM_FILE)
        except Exception:
            st.warning("Incremental LSTM training failed this round.")

    # record prediction vs actual
    predictions = {"num": pred_num, "col": pred_col_from_argmax, "size": pred_size, "confidences": confidences}
    record_confirmation(predictions, actual_entry)
    st.success("Confirmed & learned — models/prior updated.")

# Downloads & history
st.markdown("---")
st.subheader("History & logs (downloadable)")

if st.session_state.history:
    df_hist = pd.DataFrame([{"Round": i+1,
                             "Number": h["num"],
                             "Color": h["color"],
                             "Size": "B" if h.get("size_observed", (h["num"]>=5))==1 else "S"} for i,h in enumerate(st.session_state.history)])
    st.dataframe(df_hist.tail(100))
    buf = BytesIO()
    df_hist.to_excel(buf, index=False)
    st.download_button("⬇️ Download history (Excel)", data=buf.getvalue(), file_name="history_no_violet_improved.xlsx")

if st.session_state.log:
    df_log = pd.DataFrame(st.session_state.log)
    st.dataframe(df_log.tail(200))
    buf2 = BytesIO()
    df_log.to_excel(buf2, index=False)
    st.download_button("⬇️ Download prediction log (Excel)", data=buf2.getvalue(), file_name="prediction_log_no_violet_improved.xlsx")

# Accuracy graph & stats
st.markdown("---")
st.subheader("Accuracy & stats")
plot_accuracy_history()
num_conf_avg = np.mean([l["conf_num"] for l in st.session_state.log]) if st.session_state.log else 0.0
st.write(f"Average predicted-number confidence (history): {num_conf_avg:.1f}%")
st.write(f"Loss streak (consecutive incorrect numbers): {st.session_state.loss_streak}")

st.caption("Notes: Primary color = color of the most probable (argmax) number. Aggregated odd/even color mass is shown for diagnostics. Ensemble = LSTM + Markov + NaiveBayes for number prediction.")

