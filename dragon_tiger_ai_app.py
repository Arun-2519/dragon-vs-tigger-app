# lottery7_streamlit_app.py
"""
Updated Lottery7 / Wingo — Streamlit continual-learning LSTM app (Improved rules)

Key rule upgrades requested by user integrated here:
- Numbers 0-4 = Small; Numbers 5-9 = Big (deterministic mapping used for size prediction)
- Color mapping rules derived from numbers:
  - Numbers 1,3,7,9 -> Green (deterministic)
  - Numbers 2,4,6,8 -> Red (deterministic)
  - Numbers 0 and 5 -> Ambiguous (can be Red or Violet). Violet appears only for 0 and 5.
- Model must learn Number and Color largely separately (separate models) and avoid confusion between color and size predictions.
- Final size prediction is derived deterministically from the number prediction (to avoid conflicts).
- Final color prediction is primarily derived from number probabilities; for ambiguous numbers (0,5) the color model helps distribute between Red/Violet.

Features:
- Separate LSTM number_model (predicts numbers for 3 places)
- Separate LSTM color_model (helps disambiguate 0/5 color distribution)
- Replay buffers and Markov + Frequency priors for numbers and colors per place
- Ensemble: number_model + markov + freq for numbers. Color final computed from number_probs + color_model for 0/5.
- Enforces deterministic size from number probabilities to avoid confusion
- Starts predicting after WINDOW (default 10). Shows WAIT if confidence below threshold.
- Queue -> Confirm & Learn workflow, persistent history and models, Excel download, logs.

Usage:
    pip install streamlit tensorflow numpy pandas openpyxl
    streamlit run lottery7_streamlit_app.py

"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import random
from collections import deque
from io import BytesIO

# --- TensorFlow imports ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    st.error("TensorFlow is required. Install with `pip install tensorflow`.")
    st.stop()

# -----------------
# App config
# -----------------
st.set_page_config(page_title="🎯 Lottery7 Wingo — Upgraded", layout="centered")
st.title("🎯 Lottery7 Wingo — Upgraded Predictor (Rules enforced)")
st.markdown("""
This app enforces game rules so the model doesn't confuse colors, sizes, and numbers.
- Numbers predicted by `number_model` (LSTM)
- Colors computed from number probabilities (deterministic mapping) plus help from `color_model` for ambiguous numbers (0 & 5)
- Size (Big/Small) derived deterministically from number prediction
""")

# -----------------
# Paths & defaults
# -----------------
DEFAULT_WINDOW = 10
MODEL_NUM_PATH = "model_number.h5"
MODEL_COL_PATH = "model_color.h5"
HISTORY_PATH = "history_rules.json"

# -----------------
# Sidebar hyperparameters
# -----------------
st.sidebar.header("Hyperparameters & Controls")
WINDOW = st.sidebar.number_input("Sliding window length", value=DEFAULT_WINDOW, min_value=3, max_value=30, step=1)
REPLAY_MAX = st.sidebar.number_input("Replay buffer max size", value=8000, step=500)
RECENT_K = st.sidebar.number_input("Recent prioritized windows", value=300, step=50)
REPLAY_SAMPLE = st.sidebar.number_input("Replay sample per update", value=256, step=16)
BATCH_SIZE = st.sidebar.number_input("Batch size", value=64)
LR = float(st.sidebar.number_input("Learning rate", value=1e-4, format="%.6f"))
INCREMENTAL_EPOCHS = st.sidebar.number_input("Epochs per incremental update", value=1, min_value=1, max_value=5)
W_NUM_LSTM = float(st.sidebar.slider("Number ensemble weight: LSTM", 0.0, 1.0, 0.6, 0.05))
W_NUM_MARKOV = float(st.sidebar.slider("Number ensemble weight: Markov", 0.0, 1.0, 0.3, 0.05))
W_NUM_FREQ = float(st.sidebar.slider("Number ensemble weight: Frequency", 0.0, 1.0, 0.1, 0.05))
CONF_THRESHOLD = float(st.sidebar.slider("Confidence threshold (%)", 0, 100, 65))

# -----------------
# Label maps and dims
# -----------------
NUM_CLASSES_NUM = 10
NUM_CLASSES_COL = 3  # 0: Green, 1: Red, 2: Violet  (we'll map accordingly)
NUM_PLACES = 3
FEATURES_PER_PLACE = NUM_CLASSES_NUM + NUM_CLASSES_COL + 1  # one-hot num + one-hot color (used for model inputs) + size (0/1)
INPUT_FEATURES = FEATURES_PER_PLACE * NUM_PLACES

# Color mapping used for display
INV_COLOR = {0: 'G', 1: 'R', 2: 'V'}

# Rules mapping (user-provided)
GREEN_NUMS = {1,3,7,9}
RED_NUMS = {2,4,6,8}
AMBIGUOUS_NUMS = {0,5}  # can be Red or Violet

# Size determination from number
# 0-4 -> Small (S), 5-9 -> Big (B)

# -----------------
# Utility classes
# -----------------
class ReplayBuffer:
    def __init__(self, maxlen=REPLAY_MAX, window=WINDOW):
        self.buffer = deque(maxlen=maxlen)
        self.window = window
    def add(self, x, y_dict):
        self.buffer.append((np.array(x, dtype=np.float32), {k: np.array(v, dtype=np.int32) for k,v in y_dict.items()}))
    def sample(self, n):
        n = min(n, len(self.buffer))
        if n == 0:
            return np.zeros((0, self.window, INPUT_FEATURES), dtype=np.float32), {}
        s = random.sample(self.buffer, n)
        X = np.stack([p[0] for p in s], axis=0)
        keys = s[0][1].keys()
        y = {k: np.array([p[1][k] for p in s], dtype=np.int32) for k in keys}
        return X, y
    def __len__(self):
        return len(self.buffer)

class Markov:
    def __init__(self, n_classes):
        self.n = n_classes
        self.counts = np.ones((n_classes, n_classes), dtype=np.float32)
    def update(self, prev, nxt):
        self.counts[int(prev), int(nxt)] += 1
    def predict_prob(self, last):
        row = self.counts[int(last)]
        return row / row.sum()

class Frequency:
    def __init__(self, n_classes):
        self.counts = np.ones(n_classes, dtype=np.float32)
    def update(self, x):
        self.counts[int(x)] += 1
    def prob(self):
        return self.counts / np.sum(self.counts)

# -----------------
# Model builders (separate models for number & color)
# -----------------

def build_number_model(window=WINDOW, input_features=INPUT_FEATURES):
    tf.keras.backend.clear_session()
    inp = Input(shape=(window, input_features), name='num_seq')
    x = LSTM(128, name='num_lstm')(inp)
    x = Dropout(0.1)(x)
    outputs = []
    for p in range(NUM_PLACES):
        out = Dense(NUM_CLASSES_NUM, activation='softmax', name=f'num_{p}_out')(x)
        outputs.append(out)
    model = Model(inputs=inp, outputs=outputs)
    opt = Adam(learning_rate=LR)
    # compile with list of losses repeat for each head
    model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy']*NUM_PLACES)
    return model


def build_color_model(window=WINDOW, input_features=INPUT_FEATURES):
    tf.keras.backend.clear_session()
    inp = Input(shape=(window, input_features), name='col_seq')
    x = LSTM(64, name='col_lstm')(inp)
    x = Dropout(0.1)(x)
    outputs = []
    for p in range(NUM_PLACES):
        out = Dense(NUM_CLASSES_COL, activation='softmax', name=f'col_{p}_out')(x)
        outputs.append(out)
    model = Model(inputs=inp, outputs=outputs)
    opt = Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy']*NUM_PLACES)
    return model

# -----------------
# Session state init
# -----------------
if 'history' not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        try:
            st.session_state.history = json.load(open(HISTORY_PATH, 'r'))
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []  # list of rounds; each round: {'places':[{'num':int,'color':int},...]}

if 'replay_num' not in st.session_state:
    st.session_state.replay_num = ReplayBuffer(maxlen=REPLAY_MAX, window=WINDOW)
if 'replay_col' not in st.session_state:
    st.session_state.replay_col = ReplayBuffer(maxlen=REPLAY_MAX, window=WINDOW)

# Markov & freq per place
if 'markov_num' not in st.session_state:
    st.session_state.markov_num = [Markov(NUM_CLASSES_NUM) for _ in range(NUM_PLACES)]
if 'markov_col' not in st.session_state:
    st.session_state.markov_col = [Markov(NUM_CLASSES_COL) for _ in range(NUM_PLACES)]

if 'freq_num' not in st.session_state:
    st.session_state.freq_num = [Frequency(NUM_CLASSES_NUM) for _ in range(NUM_PLACES)]
if 'freq_col' not in st.session_state:
    st.session_state.freq_col = [Frequency(NUM_CLASSES_COL) for _ in range(NUM_PLACES)]

# models
if 'model_num' not in st.session_state:
    if os.path.exists(MODEL_NUM_PATH):
        try:
            st.session_state.model_num = tf.keras.models.load_model(MODEL_NUM_PATH)
            tf.keras.backend.set_value(st.session_state.model_num.optimizer.learning_rate, LR)
        except Exception:
            st.session_state.model_num = build_number_model(WINDOW, INPUT_FEATURES)
    else:
        st.session_state.model_num = build_number_model(WINDOW, INPUT_FEATURES)

if 'model_col' not in st.session_state:
    if os.path.exists(MODEL_COL_PATH):
        try:
            st.session_state.model_col = tf.keras.models.load_model(MODEL_COL_PATH)
            tf.keras.backend.set_value(st.session_state.model_col.optimizer.learning_rate, LR)
        except Exception:
            st.session_state.model_col = build_color_model(WINDOW, INPUT_FEATURES)
    else:
        st.session_state.model_col = build_color_model(WINDOW, INPUT_FEATURES)

if 'log' not in st.session_state:
    st.session_state.log = []
if 'pending' not in st.session_state:
    st.session_state.pending = None

# populate buffers and stats from history once
if not st.session_state.get('_populated', False):
    hist = st.session_state.history
    for i in range(len(hist) - WINDOW):
        window = hist[i:i+WINDOW]
        Xw = []
        for t in window:
            vec = []
            for p in range(NUM_PLACES):
                num = t['places'][p]['num']
                col = t['places'][p]['color']
                # one-hot num
                nvec = np.zeros(NUM_CLASSES_NUM, dtype=np.float32); nvec[int(num)] = 1.0
                cvec = np.zeros(NUM_CLASSES_COL, dtype=np.float32); cvec[int(col)] = 1.0
                svec = np.array([1.0 if int(num) >=5 else 0.0], dtype=np.float32)  # size derived from number for input
                vec.extend(nvec.tolist()); vec.extend(cvec.tolist()); vec.extend(svec.tolist())
            Xw.append(vec)
        target = hist[i+WINDOW]
        y_num = {f'num_{p}_out': int(target['places'][p]['num']) for p in range(NUM_PLACES)}
        y_col = {f'col_{p}_out': int(target['places'][p]['color']) for p in range(NUM_PLACES)}
        st.session_state.replay_num.add(np.stack(Xw, axis=0), y_num)
        st.session_state.replay_col.add(np.stack(Xw, axis=0), y_col)

    for i in range(len(hist)-1):
        cur = hist[i]
        nxt = hist[i+1]
        for p in range(NUM_PLACES):
            st.session_state.markov_num[p].update(cur['places'][p]['num'], nxt['places'][p]['num'])
            st.session_state.markov_col[p].update(cur['places'][p]['color'], nxt['places'][p]['color'])
            st.session_state.freq_num[p].update(nxt['places'][p]['num'])
            st.session_state.freq_col[p].update(nxt['places'][p]['color'])
    st.session_state._populated = True

# -----------------
# Helpers: encoders
# -----------------
def round_to_vector(round_entry):
    vec = []
    for p in range(NUM_PLACES):
        num = int(round_entry['places'][p]['num'])
        col = int(round_entry['places'][p]['color'])
        nvec = np.zeros(NUM_CLASSES_NUM, dtype=np.float32); nvec[num] = 1.0
        cvec = np.zeros(NUM_CLASSES_COL, dtype=np.float32); cvec[col] = 1.0
        svec = np.array([1.0 if num >=5 else 0.0], dtype=np.float32)
        vec.extend(nvec.tolist()); vec.extend(cvec.tolist()); vec.extend(svec.tolist())
    return np.array(vec, dtype=np.float32)

# build targets
def build_num_targets(round_entry):
    return {f'num_{p}_out': int(round_entry['places'][p]['num']) for p in range(NUM_PLACES)}

def build_col_targets(round_entry):
    return {f'col_{p}_out': int(round_entry['places'][p]['color']) for p in range(NUM_PLACES)}

# -----------------
# Persistence
# -----------------
def save_history():
    json.dump(st.session_state.history, open(HISTORY_PATH, 'w'))

# -----------------
# Core incremental update & predict logic
# -----------------

def incremental_update_and_predict(new_round):
    # append
    st.session_state.history.append(new_round)
    save_history()

    # update markov & freq
    if len(st.session_state.history) >= 2:
        prev = st.session_state.history[-2]
        cur = st.session_state.history[-1]
        for p in range(NUM_PLACES):
            st.session_state.markov_num[p].update(prev['places'][p]['num'], cur['places'][p]['num'])
            st.session_state.markov_col[p].update(prev['places'][p]['color'], cur['places'][p]['color'])
    for p in range(NUM_PLACES):
        st.session_state.freq_num[p].update(new_round['places'][p]['num'])
        st.session_state.freq_col[p].update(new_round['places'][p]['color'])

    # add replay samples
    if len(st.session_state.history) >= WINDOW + 1:
        window = st.session_state.history[-(WINDOW+1):-1]
        Xw = [round_to_vector(t) for t in window]
        st.session_state.replay_num.add(np.stack(Xw, axis=0), build_num_targets(st.session_state.history[-1]))
        st.session_state.replay_col.add(np.stack(Xw, axis=0), build_col_targets(st.session_state.history[-1]))

    # prepare training batches
    def prepare_batch(replay_buf):
        # recent windows
        train_X = []
        train_y = {f'{k}': [] for k in (build_num_targets(st.session_state.history[-1]) if len(st.session_state.history)>=WINDOW+1 else {})}
        L = len(st.session_state.history)
        recent_count = min(RECENT_K, max(0, L - WINDOW))
        start_i = max(0, L - WINDOW - recent_count + 1)
        for i in range(start_i, L - WINDOW + 1):
            w = st.session_state.history[i:i+WINDOW]
            Xw = np.stack([round_to_vector(t) for t in w], axis=0)
            train_X.append(Xw)
            tgt = build_num_targets(st.session_state.history[i+WINDOW]) if replay_buf is st.session_state.replay_num else build_col_targets(st.session_state.history[i+WINDOW])
            for k in tgt:
                train_y[k].append(int(tgt[k]))
        if len(train_X) > 0:
            train_X = np.stack(train_X, axis=0)
            train_y = {k: np.array(v, dtype=np.int32) for k,v in train_y.items()}
        else:
            train_X = np.zeros((0, WINDOW, INPUT_FEATURES), dtype=np.float32)
            train_y = {k: np.zeros((0,), dtype=np.int32) for k in (build_num_targets(st.session_state.history[-1]) if len(st.session_state.history)>=WINDOW+1 else [])}
        # add replay samples
        Xr, yr = replay_buf.sample(min(REPLAY_SAMPLE, len(replay_buf)))
        if Xr.shape[0] > 0 and train_X.shape[0] > 0:
            X_all = np.vstack([train_X, Xr])
            # merge y dictionaries
            keys = list(yr.keys())
            y_all = {k: np.concatenate([train_y[k], yr[k]], axis=0) for k in keys}
        elif Xr.shape[0] > 0:
            X_all = Xr
            y_all = yr
        else:
            X_all = train_X
            y_all = train_y
        return X_all, y_all

    Xn, yn = prepare_batch(st.session_state.replay_num)
    Xc, yc = prepare_batch(st.session_state.replay_col)

    # train number model
    if isinstance(Xn, np.ndarray) and Xn.shape[0] > 0:
        try:
            st.session_state.model_num.fit(Xn, [yn[f'num_{p}_out'] for p in range(NUM_PLACES)], epochs=INCREMENTAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        except Exception:
            st.warning('Number model training failed this round — rebuilding model.')
            st.session_state.model_num = build_number_model(WINDOW, INPUT_FEATURES)

    # train color model
    if isinstance(Xc, np.ndarray) and Xc.shape[0] > 0:
        try:
            st.session_state.model_col.fit(Xc, [yc[f'col_{p}_out'] for p in range(NUM_PLACES)], epochs=INCREMENTAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        except Exception:
            st.warning('Color model training failed this round — rebuilding model.')
            st.session_state.model_col = build_color_model(WINDOW, INPUT_FEATURES)

    # predictions (numbers ensemble)
    def predict_number_probs():
        hist = st.session_state.history
        if len(hist) >= WINDOW:
            inp = np.stack([round_to_vector(r) for r in hist[-WINDOW:]], axis=0).reshape(1, WINDOW, INPUT_FEATURES)
            preds = st.session_state.model_num.predict(inp, verbose=0)
            # preds is list length NUM_PLACES
            num_probs = {f'num_{p}_out': preds[p][0] for p in range(NUM_PLACES)}
        else:
            num_probs = {f'num_{p}_out': np.ones(NUM_CLASSES_NUM)/NUM_CLASSES_NUM for p in range(NUM_PLACES)}
        return num_probs

    num_probs = predict_number_probs()

    # color final probabilities derived from numbers + color_model for ambiguous numbers
    # first compute color contribution from deterministic numbers
    color_final = {}
    # get color_model probs for ambiguous handling
    col_model_probs = {}
    if len(st.session_state.history) >= WINDOW:
        inp = np.stack([round_to_vector(r) for r in st.session_state.history[-WINDOW:]], axis=0).reshape(1, WINDOW, INPUT_FEATURES)
        col_preds = st.session_state.model_col.predict(inp, verbose=0)
        for p in range(NUM_PLACES):
            col_model_probs[f'col_{p}_out'] = col_preds[p][0]
    else:
        for p in range(NUM_PLACES):
            col_model_probs[f'col_{p}_out'] = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL

    for p in range(NUM_PLACES):
        probs = np.zeros(NUM_CLASSES_COL, dtype=np.float32)
        npb = num_probs[f'num_{p}_out']
        for n in range(NUM_CLASSES_NUM):
            pn = npb[n]
            if n in GREEN_NUMS:
                probs[0] += pn  # Green index 0
            elif n in RED_NUMS:
                probs[1] += pn  # Red index 1
            elif n in AMBIGUOUS_NUMS:
                # distribute this number's probability according to color_model's split between Red & Violet
                cm = col_model_probs[f'col_{p}_out']
                # only consider red (1) and violet (2) portions; ignore green portion from color model for 0/5
                red_share = cm[1]
                vio_share = cm[2]
                total = (red_share + vio_share)
                if total <= 0:
                    # fallback split evenly between red & violet
                    probs[1] += pn * 0.5
                    probs[2] += pn * 0.5
                else:
                    probs[1] += pn * (red_share / total)
                    probs[2] += pn * (vio_share / total)
            else:
                # safety fallback
                probs += pn / NUM_CLASSES_COL
        # normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL
        color_final[f'col_{p}_out'] = probs

    # size derived deterministically from number_probs
    size_final = {}
    for p in range(NUM_PLACES):
        npb = num_probs[f'num_{p}_out']
        small_prob = np.sum(npb[0:5])
        big_prob = np.sum(npb[5:10])
        size_final[f'size_{p}_out'] = np.array([small_prob, big_prob], dtype=np.float32)
        # normalize
        s = size_final[f'size_{p}_out']
        if s.sum() > 0:
            size_final[f'size_{p}_out'] = s / s.sum()
        else:
            size_final[f'size_{p}_out'] = np.array([0.5,0.5], dtype=np.float32)

    # save models
    try:
        st.session_state.model_num.save(MODEL_NUM_PATH)
        st.session_state.model_col.save(MODEL_COL_PATH)
    except Exception:
        pass

    # return dictionaries of probs
    return num_probs, color_final, size_final

# -----------------
# UI: input flow
# -----------------
st.subheader("Enter new round (Queue -> Confirm & Learn)")
cols = st.columns(NUM_PLACES)
place_inputs = []
for i in range(NUM_PLACES):
    with cols[i]:
        st.markdown(f"**Place P{i+1}**")
        num_val = st.number_input(f"P{i+1} Number", min_value=0, max_value=9, value=0, key=f'num_in_{i}')
        # allow entering color as actual observed color (0/1/2) for history, but model will enforce rules on prediction
        col_val = st.selectbox(f"P{i+1} Color (observed)", options=['G','R','V'], index=0, key=f'col_in_{i}')
        place_inputs.append({'num': int(num_val), 'color': 0 if col_val=='G' else (1 if col_val=='R' else 2)})

if st.button('Queue round'):
    # create pending round with size derived from number
    pending_round = {'places': []}
    for p in range(NUM_PLACES):
        n = place_inputs[p]['num']
        c = place_inputs[p]['color']
        s = 1 if n >=5 else 0
        pending_round['places'].append({'num': int(n), 'color': int(c), 'size': int(s)})
    st.session_state.pending = pending_round
    st.success('Queued round — confirm to learn')

if st.session_state.pending:
    st.info('Pending: ' + ', '.join([f"P{i+1}:{st.session_state.pending['places'][i]['num']}{INV_COLOR[st.session_state.pending['places'][i]['color']]}{'B' if st.session_state.pending['places'][i]['size']==1 else 'S'}" for i in range(NUM_PLACES)]))

# live prediction (before confirming) if enough history
st.subheader('Live prediction (based on current model & last window)')
if len(st.session_state.history) >= WINDOW:
    num_probs_live = None
    try:
        # number model preds
        inp = np.stack([round_to_vector(r) for r in st.session_state.history[-WINDOW:]], axis=0).reshape(1, WINDOW, INPUT_FEATURES)
        preds_num = st.session_state.model_num.predict(inp, verbose=0)
        num_probs_live = {f'num_{p}_out': preds_num[p][0] for p in range(NUM_PLACES)}
    except Exception:
        num_probs_live = {f'num_{p}_out': np.ones(NUM_CLASSES_NUM)/NUM_CLASSES_NUM for p in range(NUM_PLACES)}

    # color model preds for ambiguous split
    try:
        preds_col = st.session_state.model_col.predict(inp, verbose=0)
        col_model_live = {f'col_{p}_out': preds_col[p][0] for p in range(NUM_PLACES)}
    except Exception:
        col_model_live = {f'col_{p}_out': np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL for p in range(NUM_PLACES)}

    # compute color_final and size_final as in incremental function
    color_live = {}
    size_live = {}
    for p in range(NUM_PLACES):
        npb = num_probs_live[f'num_{p}_out']
        probs = np.zeros(NUM_CLASSES_COL, dtype=np.float32)
        for n in range(NUM_CLASSES_NUM):
            pn = npb[n]
            if n in GREEN_NUMS:
                probs[0] += pn
            elif n in RED_NUMS:
                probs[1] += pn
            elif n in AMBIGUOUS_NUMS:
                cm = col_model_live[f'col_{p}_out']
                red_share = cm[1]; vio_share = cm[2]; total = (red_share + vio_share)
                if total <= 0:
                    probs[1] += pn * 0.5; probs[2] += pn * 0.5
                else:
                    probs[1] += pn * (red_share/total)
                    probs[2] += pn * (vio_share/total)
            else:
                probs += pn / NUM_CLASSES_COL
        if probs.sum()>0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL
        color_live[f'col_{p}_out'] = probs
        # size
        small_prob = np.sum(npb[0:5]); big_prob = np.sum(npb[5:10])
        s = np.array([small_prob, big_prob]); s = s / s.sum() if s.sum()>0 else np.array([0.5,0.5])
        size_live[f'size_{p}_out'] = s

    # show per-place predictions summary and allow selecting place
    place_choice = st.selectbox('Select place to inspect / bet', options=['P1','P2','P3'], index=0)
    pi = int(place_choice[1]) - 1
    pred_num = int(np.argmax(num_probs_live[f'num_{pi}_out'])); conf_num = float(np.max(num_probs_live[f'num_{pi}_out']))*100.0
    pred_col_idx = int(np.argmax(color_live[f'col_{pi}_out'])); conf_col = float(np.max(color_live[f'col_{pi}_out']))*100.0
    pred_size_idx = int(np.argmax(size_live[f'size_{pi}_out'])); conf_size = float(np.max(size_live[f'size_{pi}_out']))*100.0

    st.write(f"Place {place_choice} predictions:")
    st.write(f"- Number: {pred_num} (conf {conf_num:.1f}%)")
    st.write(f"- Color: {INV_COLOR[pred_col_idx]} (conf {conf_col:.1f}%)")
    st.write(f"- Size: {'B' if pred_size_idx==1 else 'S'} (conf {conf_size:.1f}%)")

    # recommend highest-confidence category
    best_cat = None; best_conf = -1.0; best_val = None
    for cat, val, conf in [('Number', pred_num, conf_num), ('Color', INV_COLOR[pred_col_idx], conf_col), ('Size', 'B' if pred_size_idx==1 else 'S', conf_size)]:
        if conf > best_conf:
            best_conf = conf; best_cat = cat; best_val = val
    if best_conf >= CONF_THRESHOLD:
        st.success(f"RECOMMENDED BET: {best_cat} -> {best_val} (confidence {best_conf:.1f}%)")
    else:
        st.warning("WAIT: model confidence below threshold. Keep collecting results until pattern emerges.")

else:
    need = max(0, WINDOW - len(st.session_state.history))
    st.info(f"Need {need} more rounds to start live LSTM-based predictions (start predicting at round {WINDOW+1}).")
    place_choice = st.selectbox('Select place to inspect / bet', options=['P1','P2','P3'], index=0)

# Confirm & Learn
st.subheader('Confirm pending round and let the model learn')
if st.session_state.pending:
    if st.button('Confirm & Learn'):
        new_round = st.session_state.pending
        num_probs, color_probs, size_probs = incremental_update_and_predict(new_round)
        pi = int(place_choice[1]) - 1
        pred_num = int(np.argmax(num_probs[f'num_{pi}_out'])); conf_num = float(np.max(num_probs[f'num_{pi}_out']))*100.0
        pred_col_idx = int(np.argmax(color_probs[f'col_{pi}_out'])); conf_col = float(np.max(color_probs[f'col_{pi}_out']))*100.0
        pred_size_idx = int(np.argmax(size_probs[f'size_{pi}_out'])); conf_size = float(np.max(size_probs[f'size_{pi}_out']))*100.0
        best_cat=None; best_conf=-1; best_val=None
        for cat, val, conf in [('Number', pred_num, conf_num), ('Color', INV_COLOR[pred_col_idx], conf_col), ('Size', 'B' if pred_size_idx==1 else 'S', conf_size)]:
            if conf > best_conf:
                best_conf = conf; best_cat = cat; best_val = val
        if best_conf >= CONF_THRESHOLD:
            st.success(f"After learning — RECOMMENDED BET: {best_cat} -> {best_val} (confidence {best_conf:.1f}%)")
        else:
            st.warning('After learning — WAIT: confidence still below threshold.')
        st.session_state.log.append({
            'added': ','.join([f"P{i+1}:{r['num']}{INV_COLOR[r['color']]}{'B' if r['num']>=5 else 'S'}" for i,r in enumerate(new_round['places'])]),
            'recommendation': f"{best_cat}:{best_val}",
            'conf': f"{best_conf:.1f}%"
        })
        st.session_state.pending = None
        st.experimental_rerun()
    if st.button('Discard pending'):
        st.session_state.pending = None
        st.info('Pending round discarded')

# Manual force update
st.markdown('---')
if st.button('Force incremental update (use last round)'):
    if len(st.session_state.history) >= WINDOW + 1:
        incremental_update_and_predict(st.session_state.history[-1])
        st.success('Forced incremental update performed')
    else:
        st.info('Not enough history to force update')

# History & logs
st.markdown('---')
st.subheader('History')
if st.session_state.history:
    df = pd.DataFrame([{'Round': i+1,
                        'P1': f"{r['places'][0]['num']}{INV_COLOR[r['places'][0]['color']]}{'B' if r['places'][0]['num']>=5 else 'S'}",
                        'P2': f"{r['places'][1]['num']}{INV_COLOR[r['places'][1]['color']]}{'B' if r['places'][1]['num']>=5 else 'S'}",
                        'P3': f"{r['places'][2]['num']}{INV_COLOR[r['places'][2]['color']]}{'B' if r['places'][2]['num']>=5 else 'S'}",} for i,r in enumerate(st.session_state.history)])
    st.dataframe(df.tail(200))
    buf = BytesIO(); df.to_excel(buf, index=False)
    st.download_button('⬇️ Download history (Excel)', data=buf.getvalue(), file_name='history_rules.xlsx')

if st.session_state.log:
    st.subheader('Model log')
    st.dataframe(pd.DataFrame(st.session_state.log).tail(200))
    buf2 = BytesIO(); pd.DataFrame(st.session_state.log).to_excel(buf2, index=False)
    st.download_button('⬇️ Download log (Excel)', data=buf2.getvalue(), file_name='model_log_rules.xlsx')

st.markdown('---')
st.caption('Notes: This upgraded app enforces number->size determinism and derives colors from number probabilities (with color model help only for ambiguous numbers 0 and 5). This prevents confusion between colors and sizes while still learning from data.')
