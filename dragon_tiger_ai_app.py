# lottery7_streamlit_deep.py
"""
Lottery7 Wingo — Streamlit app with incremental LSTM number & color models
Minimal UI: 3 inputs per place (Size, Color, Number)
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, random
from collections import deque
from io import BytesIO

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    st.error("TensorFlow is required. Install with `pip install tensorflow`.")
    st.stop()

# ----------------------
# Config / rules
# ----------------------
st.set_page_config(page_title="Lottery7 Deep Predictor", layout="centered")
st.title("🎯 Lottery7 Wingo — Deep LSTM Predictor (Minimal UI)")

DEFAULT_WINDOW = 10
MODEL_NUM_PATH = "model_number.h5"
MODEL_COL_PATH = "model_color.h5"
HISTORY_PATH = "history_rules.json"

# Sidebar hyperparameters
st.sidebar.header("Hyperparameters")
WINDOW = st.sidebar.number_input("Window length", value=DEFAULT_WINDOW, min_value=3, max_value=30)
BATCH_SIZE = st.sidebar.number_input("Batch size", value=32)
LR = float(st.sidebar.number_input("Learning rate", value=1e-4, format="%.6f"))
INCREMENTAL_EPOCHS = st.sidebar.number_input("Epochs per update", value=1, min_value=1, max_value=5)
CONF_THRESHOLD = st.sidebar.slider("Confidence threshold %", 0, 100, 65)

# dims
NUM_PLACES = 3
NUM_CLASSES_NUM = 10
NUM_CLASSES_COL = 3  # G=0, R=1, V=2
FEATURES_PER_PLACE = NUM_CLASSES_NUM + NUM_CLASSES_COL + 1  # one-hot num + one-hot col + size bit
INPUT_FEATURES = FEATURES_PER_PLACE * NUM_PLACES

INV_COLOR = {0: 'G', 1: 'R', 2: 'V'}
GREEN_NUMS = {1,3,7,9}
RED_NUMS = {2,4,6,8}
AMBIGUOUS_NUMS = {0,5}

# ----------------------
# Simple replay buffer
# ----------------------
class ReplayBuffer:
    def __init__(self, maxlen=5000):
        self.buff = deque(maxlen=maxlen)
    def add(self, X, y_num, y_col):
        # X: (window, features); y_num, y_col are dicts with keys num_0..num_2, col_0..col_2
        self.buff.append((np.array(X, dtype=np.float32), {**y_num, **y_col}))
    def sample(self, n):
        n = min(n, len(self.buff))
        if n == 0:
            return np.zeros((0, WINDOW, INPUT_FEATURES), dtype=np.float32), {}
        s = random.sample(list(self.buff), n)
        X = np.stack([p[0] for p in s], axis=0)
        keys = s[0][1].keys()
        y = {k: np.array([p[1][k] for p in s], dtype=np.int32) for k in keys}
        return X, y
    def __len__(self):
        return len(self.buff)

# ----------------------
# Build models
# ----------------------
def build_number_model():
    tf.keras.backend.clear_session()
    inp = Input(shape=(WINDOW, INPUT_FEATURES), name='num_input')
    x = LSTM(128, name='num_lstm')(inp)
    x = Dropout(0.1)(x)
    outs = [Dense(NUM_CLASSES_NUM, activation='softmax', name=f'num_{p}_out')(x) for p in range(NUM_PLACES)]
    m = Model(inputs=inp, outputs=outs)
    m.compile(optimizer=Adam(LR), loss=['sparse_categorical_crossentropy']*NUM_PLACES)
    return m

def build_color_model():
    tf.keras.backend.clear_session()
    inp = Input(shape=(WINDOW, INPUT_FEATURES), name='col_input')
    x = LSTM(64, name='col_lstm')(inp)
    x = Dropout(0.1)(x)
    outs = [Dense(NUM_CLASSES_COL, activation='softmax', name=f'col_{p}_out')(x) for p in range(NUM_PLACES)]
    m = Model(inputs=inp, outputs=outs)
    m.compile(optimizer=Adam(LR), loss=['sparse_categorical_crossentropy']*NUM_PLACES)
    return m

# ----------------------
# Session state init
# ----------------------
if 'history' not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        try:
            st.session_state.history = json.load(open(HISTORY_PATH, 'r'))
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

if 'replay' not in st.session_state:
    st.session_state.replay = ReplayBuffer(maxlen=5000)

if 'markov' not in st.session_state:
    # simple Markov for numbers per place
    st.session_state.markov = [np.ones((NUM_CLASSES_NUM, NUM_CLASSES_NUM), dtype=np.float32) for _ in range(NUM_PLACES)]

if 'freq_num' not in st.session_state:
    st.session_state.freq_num = [np.ones(NUM_CLASSES_NUM, dtype=np.float32) for _ in range(NUM_PLACES)]

if 'freq_col' not in st.session_state:
    st.session_state.freq_col = [np.ones(NUM_CLASSES_COL, dtype=np.float32) for _ in range(NUM_PLACES)]

if 'model_num' not in st.session_state:
    if os.path.exists(MODEL_NUM_PATH):
        try:
            st.session_state.model_num = tf.keras.models.load_model(MODEL_NUM_PATH)
            tf.keras.backend.set_value(st.session_state.model_num.optimizer.learning_rate, LR)
        except Exception:
            st.session_state.model_num = build_number_model()
    else:
        st.session_state.model_num = build_number_model()

if 'model_col' not in st.session_state:
    if os.path.exists(MODEL_COL_PATH):
        try:
            st.session_state.model_col = tf.keras.models.load_model(MODEL_COL_PATH)
            tf.keras.backend.set_value(st.session_state.model_col.optimizer.learning_rate, LR)
        except Exception:
            st.session_state.model_col = build_color_model()
    else:
        st.session_state.model_col = build_color_model()

if 'pending' not in st.session_state:
    st.session_state.pending = None
if 'log' not in st.session_state:
    st.session_state.log = []

# populate replay from history if present (one-time)
if not st.session_state.get('_populated', False):
    hist = st.session_state.history
    for i in range(len(hist) - WINDOW):
        win = hist[i:i+WINDOW]
        Xw = [None]*WINDOW
        for t_idx, t in enumerate(win):
            vec = []
            for p in range(NUM_PLACES):
                n = int(t['places'][p]['num'])
                c = int(t['places'][p]['color'])
                nvec = np.zeros(NUM_CLASSES_NUM); nvec[n] = 1.0
                cvec = np.zeros(NUM_CLASSES_COL); cvec[c] = 1.0
                svec = np.array([1.0 if n >=5 else 0.0])
                vec.extend(nvec.tolist()); vec.extend(cvec.tolist()); vec.extend(svec.tolist())
            Xw[t_idx] = vec
        target = hist[i+WINDOW]
        y_num = {f'num_{p}_out': int(target['places'][p]['num']) for p in range(NUM_PLACES)}
        y_col = {f'col_{p}_out': int(target['places'][p]['color']) for p in range(NUM_PLACES)}
        st.session_state.replay.add(np.stack(Xw, axis=0), y_num, y_col)
    # markov/freq
    for i in range(len(hist)-1):
        cur = hist[i]; nxt = hist[i+1]
        for p in range(NUM_PLACES):
            st.session_state.markov[p][cur['places'][p]['num'], nxt['places'][p]['num']] += 1
            st.session_state.freq_num[p][nxt['places'][p]['num']] += 1
            st.session_state.freq_col[p][nxt['places'][p]['color']] += 1
    st.session_state._populated = True

# ----------------------
# Helpers
# ----------------------
def round_to_vector(round_entry):
    vec = []
    for p in range(NUM_PLACES):
        n = int(round_entry['places'][p]['num'])
        c = int(round_entry['places'][p]['color'])
        nvec = np.zeros(NUM_CLASSES_NUM, dtype=np.float32); nvec[n] = 1.0
        cvec = np.zeros(NUM_CLASSES_COL, dtype=np.float32); cvec[c] = 1.0
        svec = np.array([1.0 if n >=5 else 0.0], dtype=np.float32)
        vec.extend(nvec.tolist()); vec.extend(cvec.tolist()); vec.extend(svec.tolist())
    return np.array(vec, dtype=np.float32)

def save_history():
    with open(HISTORY_PATH, 'w') as f:
        json.dump(st.session_state.history, f, indent=2)

# ----------------------
# Core incremental update
# ----------------------
def incremental_update_and_predict(new_round):
    # append history & save
    st.session_state.history.append(new_round)
    save_history()

    # update markov/freq
    if len(st.session_state.history) >= 2:
        prev = st.session_state.history[-2]
        cur = st.session_state.history[-1]
        for p in range(NUM_PLACES):
            st.session_state.markov[p][prev['places'][p]['num'], cur['places'][p]['num']] += 1
            st.session_state.freq_num[p][cur['places'][p]['num']] += 1
            st.session_state.freq_col[p][cur['places'][p]['color']] += 1

    # add to replay buffer if possible
    if len(st.session_state.history) >= WINDOW + 1:
        window = st.session_state.history[-(WINDOW+1):-1]
        Xw = [round_to_vector(t) for t in window]
        target = st.session_state.history[-1]
        y_num = {f'num_{p}_out': int(target['places'][p]['num']) for p in range(NUM_PLACES)}
        y_col = {f'col_{p}_out': int(target['places'][p]['color']) for p in range(NUM_PLACES)}
        st.session_state.replay.add(np.stack(Xw, axis=0), y_num, y_col)

    # sample replay and incremental train
    Xr, yr = st.session_state.replay.sample(min(256, len(st.session_state.replay)))
    if Xr.shape[0] > 0:
        try:
            st.session_state.model_num.fit(Xr, [yr[f'num_{p}_out'] for p in range(NUM_PLACES)],
                                          epochs=INCREMENTAL_EPOCHS, batch_size=max(8, BATCH_SIZE), verbose=0)
        except Exception:
            st.warning("Number model training failed; rebuilding model.")
            st.session_state.model_num = build_number_model()

        try:
            st.session_state.model_col.fit(Xr, [yr[f'col_{p}_out'] for p in range(NUM_PLACES)],
                                          epochs=INCREMENTAL_EPOCHS, batch_size=max(8, BATCH_SIZE), verbose=0)
        except Exception:
            st.warning("Color model training failed; rebuilding model.")
            st.session_state.model_col = build_color_model()

    # make prediction based on last WINDOW
    if len(st.session_state.history) >= WINDOW:
        inp = np.stack([round_to_vector(r) for r in st.session_state.history[-WINDOW:]], axis=0).reshape(1, WINDOW, INPUT_FEATURES)
        preds_num = st.session_state.model_num.predict(inp, verbose=0)
        preds_col = st.session_state.model_col.predict(inp, verbose=0)
        num_probs = {f'num_{p}_out': preds_num[p][0] for p in range(NUM_PLACES)}
        col_model_probs = {f'col_{p}_out': preds_col[p][0] for p in range(NUM_PLACES)}
    else:
        num_probs = {f'num_{p}_out': np.ones(NUM_CLASSES_NUM)/NUM_CLASSES_NUM for p in range(NUM_PLACES)}
        col_model_probs = {f'col_{p}_out': np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL for p in range(NUM_PLACES)}

    # derive colors & sizes from num_probs + color model for ambiguous numbers
    color_final = {}
    size_final = {}
    for p in range(NUM_PLACES):
        npb = num_probs[f'num_{p}_out']
        probs = np.zeros(NUM_CLASSES_COL, dtype=np.float32)
        for n in range(NUM_CLASSES_NUM):
            pn = npb[n]
            if n in GREEN_NUMS:
                probs[0] += pn
            elif n in RED_NUMS:
                probs[1] += pn
            elif n in AMBIGUOUS_NUMS:
                cm = col_model_probs[f'col_{p}_out']
                red_share = cm[1]; vio_share = cm[2]; total = (red_share + vio_share)
                if total <= 0:
                    probs[1] += pn*0.5; probs[2] += pn*0.5
                else:
                    probs[1] += pn * (red_share/total)
                    probs[2] += pn * (vio_share/total)
            else:
                probs += pn / NUM_CLASSES_COL
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL
        color_final[f'col_{p}_out'] = probs

        # size from number probs
        small_prob = np.sum(npb[0:5])
        big_prob = np.sum(npb[5:10])
        s = np.array([small_prob, big_prob], dtype=np.float32)
        if s.sum() > 0:
            s = s / s.sum()
        else:
            s = np.array([0.5,0.5])
        size_final[f'size_{p}_out'] = s

    # save models
    try:
        st.session_state.model_num.save(MODEL_NUM_PATH)
        st.session_state.model_col.save(MODEL_COL_PATH)
    except Exception:
        pass

    return num_probs, color_final, size_final

# ----------------------
# Minimal 3-input UI (S/B, Color, Number)
# ----------------------
st.subheader("Enter new round — three inputs per place")
cols = st.columns(NUM_PLACES)
place_inputs = []
COLOR_CHOICES = ['G','R','V','R+V','G+V']
SIZE_CHOICES = ['S','B']

for i in range(NUM_PLACES):
    with cols[i]:
        st.markdown(f"Place P{i+1}")
        size_choice = st.selectbox(f"P{i+1} Size (S/B)", options=SIZE_CHOICES, index=0, key=f'size_in_{i}')
        color_choice = st.selectbox(f"P{i+1} Color", options=COLOR_CHOICES, index=0, key=f'col_in_{i}')
        num_choice = st.number_input(f"P{i+1} Number", min_value=0, max_value=9, value=0, key=f'num_in_{i}')
        place_inputs.append({'num': int(num_choice), 'color_raw': color_choice, 'size_raw': size_choice})

if st.button("Queue round"):
    pending = {'places': []}
    for p in range(NUM_PLACES):
        n = place_inputs[p]['num']
        c_raw = place_inputs[p]['color_raw']
        s_raw = place_inputs[p]['size_raw']
        if c_raw == 'G':
            c_int = 0; amb = None
        elif c_raw == 'R':
            c_int = 1; amb = None
        elif c_raw == 'V':
            c_int = 2; amb = None
        elif c_raw == 'R+V':
            c_int = 1; amb = 'R+V'
        elif c_raw == 'G+V':
            c_int = 0; amb = 'G+V'
        else:
            c_int = 1; amb = None
        s_int = 1 if s_raw == 'B' else 0
        pending['places'].append({
            'num': int(n),
            'color': int(c_int),
            'color_raw': c_raw,
            'ambiguous_color': amb,
            'size_observed': int(s_int),
            'size_override_used': True
        })
    st.session_state.pending = pending
    st.success("Queued round — confirm to learn")

if st.session_state.pending:
    # safe display using join and format (avoids nested f-string issues)
    pending = st.session_state.pending
    parts = []
    for i in range(NUM_PLACES):
        p = pending['places'][i]
        size_label = 'B' if p['size_observed'] == 1 else 'S'
        parts.append("{}:{}{}{}".format("P"+str(i+1), p['num'], p['color_raw'], size_label))
    st.info("Pending: " + ", ".join(parts))

# ----------------------
# Live predictions
# ----------------------
st.subheader("Live prediction")
if len(st.session_state.history) >= WINDOW:
    # safe predict without try/except too many times
    num_probs, col_probs, size_probs = incremental_update_and_predict({}) if False else (None, None, None)
    # call prediction helper without adding new round - reuse incremental logic but not appending
    # Use separate predict-only snippet:
    inp_hist_len = len(st.session_state.history)
    inp = np.stack([round_to_vector(r) for r in st.session_state.history[-WINDOW:]], axis=0).reshape(1, WINDOW, INPUT_FEATURES)
    preds_num = st.session_state.model_num.predict(inp, verbose=0)
    preds_col = st.session_state.model_col.predict(inp, verbose=0)
    num_probs_live = {f'num_{p}_out': preds_num[p][0] for p in range(NUM_PLACES)}
    col_model_live = {f'col_{p}_out': preds_col[p][0] for p in range(NUM_PLACES)}

    # derive final
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
                red_share = cm[1]; vio_share = cm[2]; total = red_share + vio_share
                if total <= 0:
                    probs[1] += pn*0.5; probs[2] += pn*0.5
                else:
                    probs[1] += pn * (red_share/total)
                    probs[2] += pn * (vio_share/total)
            else:
                probs += pn / NUM_CLASSES_COL
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL
        color_live[f'col_{p}_out'] = probs
        small_prob = np.sum(npb[0:5]); big_prob = np.sum(npb[5:10])
        s = np.array([small_prob, big_prob]); s = s / s.sum() if s.sum()>0 else np.array([0.5,0.5])
        size_live[f'size_{p}_out'] = s

    place_choice = st.selectbox("Select place", options=["P1","P2","P3"])
    pi = int(place_choice[1]) - 1
    pred_num = int(np.argmax(num_probs_live[f'num_{pi}_out'])); conf_num = float(np.max(num_probs_live[f'num_{pi}_out']))*100.0
    pred_col_idx = int(np.argmax(color_live[f'col_{pi}_out'])); conf_col = float(np.max(color_live[f'col_{pi}_out']))*100.0
    pred_size_idx = int(np.argmax(size_live[f'size_{pi}_out'])); conf_size = float(np.max(size_live[f'size_{pi}_out']))*100.0

    st.write("Prediction for {}:".format(place_choice))
    st.write("- Number: {} (conf {:.1f}%)".format(pred_num, conf_num))
    st.write("- Color: {} (conf {:.1f}%)".format(INV_COLOR[pred_col_idx], conf_col))
    st.write("- Size: {} (conf {:.1f}%)".format('B' if pred_size_idx==1 else 'S', conf_size))

    # recommend highest confidence
    best_cat = None; best_conf = -1.0; best_val = None
    for cat, val, conf in [('Number', pred_num, conf_num), ('Color', INV_COLOR[pred_col_idx], conf_col), ('Size', 'B' if pred_size_idx==1 else 'S', conf_size)]:
        if conf > best_conf:
            best_conf = conf; best_cat = cat; best_val = val
    if best_conf >= CONF_THRESHOLD:
        st.success("RECOMMENDED BET: {} -> {} (conf {:.1f}%)".format(best_cat, best_val, best_conf))
    else:
        st.warning("WAIT: confidence below threshold ({:.1f}%).".format(best_conf))
else:
    need = max(0, WINDOW - len(st.session_state.history))
    st.info("Need {} more rounds to start LSTM-based live predictions.".format(need))

# ----------------------
# Confirm & Learn
# ----------------------
st.subheader("Confirm pending round")
if st.session_state.pending:
    if st.button("Confirm & Learn"):
        new_round = st.session_state.pending
        incremental_update_and_predict(new_round)
        st.session_state.pending = None
        st.success("Confirmed & learned — models updated.")
        st.experimental_rerun()
    if st.button("Discard pending"):
        st.session_state.pending = None
        st.info("Pending discarded")

# ----------------------
# History & logs
# ----------------------
st.markdown("---")
st.subheader("History")
if st.session_state.history:
    rows = []
    for i, r in enumerate(st.session_state.history):
        rows.append({
            "Round": i+1,
            "P1": "{}{}{}".format(r['places'][0]['num'], INV_COLOR[r['places'][0]['color']], 'B' if r['places'][0].get('size_observed', r['places'][0]['num']>=5) else 'S'),
            "P2": "{}{}{}".format(r['places'][1]['num'], INV_COLOR[r['places'][1]['color']], 'B' if r['places'][1].get('size_observed', r['places'][1]['num']>=5) else 'S'),
            "P3": "{}{}{}".format(r['places'][2]['num'], INV_COLOR[r['places'][2]['color']], 'B' if r['places'][2].get('size_observed', r['places'][2]['num']>=5) else 'S')
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.tail(200))
    buf = BytesIO(); df.to_excel(buf, index=False)
    st.download_button("⬇️ Download history", data=buf.getvalue(), file_name="history_deep.xlsx")

if st.session_state.log:
    st.subheader("Log")
    st.dataframe(pd.DataFrame(st.session_state.log).tail(200))
    buf2 = BytesIO(); pd.DataFrame(st.session_state.log).to_excel(buf2, index=False)
    st.download_button("⬇️ Download log", data=buf2.getvalue(), file_name="log_deep.xlsx")

st.caption("Notes: LSTM models trained incrementally. Size predictions are derived from number probabilities to avoid color/size confusion.")
