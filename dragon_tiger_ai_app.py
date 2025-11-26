import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import random
from collections import deque
from io import BytesIO

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except:
    st.error("TensorFlow missing. Install using: pip install tensorflow")
    st.stop()

st.set_page_config(page_title="Lottery7 Wingo – Single Place", layout="centered")
st.title("🎯 Lottery7 Wingo — Single Place Deep Learning Predictor")

# CONSTANTS
NUM_CLASSES_NUM = 10
NUM_CLASSES_COL = 3   # 0=G, 1=R, 2=V
WINDOW = 10
FEATURES = NUM_CLASSES_NUM + NUM_CLASSES_COL + 1  # number one-hot + color one-hot + size
MODEL_NUM = "model_num_single.h5"
MODEL_COL = "model_col_single.h5"
HISTORY_FILE = "history_single.json"

# COLOR MAP
COLOR_MAP = {"G":0, "R":1, "V":2, "R+V":1, "G+V":0}
INV_COLOR = {0:"G", 1:"R", 2:"V"}

# HISTORY
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        st.session_state.history = json.load(open(HISTORY_FILE))
    else:
        st.session_state.history = []

# Replay Buffer
class ReplayBuf:
    def __init__(self, maxlen=5000):
        self.buf = deque(maxlen=maxlen)
    def add(self, x, y):
        self.buf.append((x,y))
    def sample(self, n):
        n=min(n,len(self.buf))
        batch=random.sample(self.buf,n)
        X=np.stack([b[0] for b in batch])
        y={"num":np.array([b[1]["num"] for b in batch]),
           "col":np.array([b[1]["col"] for b in batch])}
        return X,y

# MODELS

def build_num_model():
    inp = Input(shape=(WINDOW, FEATURES))
    x = LSTM(128)(inp)
    x = Dropout(0.1)(x)
    out = Dense(NUM_CLASSES_NUM, activation="softmax")(x)
    m = Model(inp,out)
    m.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy")
    return m

def build_col_model():
    inp = Input(shape=(WINDOW, FEATURES))
    x = LSTM(64)(inp)
    x = Dropout(0.1)(x)
    out = Dense(NUM_CLASSES_COL, activation="softmax")(x)
    m = Model(inp,out)
    m.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy")
    return m

if "num_model" not in st.session_state:
    if os.path.exists(MODEL_NUM): st.session_state.num_model = tf.keras.models.load_model(MODEL_NUM)
    else: st.session_state.num_model = build_num_model()

if "col_model" not in st.session_state:
    if os.path.exists(MODEL_COL): st.session_state.col_model = tf.keras.models.load_model(MODEL_COL)
    else: st.session_state.col_model = build_col_model()

if "replay" not in st.session_state:
    st.session_state.replay = ReplayBuf()

# ENCODERS

def encode_round(r):
    num = r["num"]
    col = COLOR_MAP[r["color"]]
    size = 1 if num>=5 else 0
    vec = []
    num_vec = np.zeros(NUM_CLASSES_NUM); num_vec[num]=1
    col_vec = np.zeros(NUM_CLASSES_COL); col_vec[col]=1
    vec.extend(num_vec); vec.extend(col_vec); vec.append(size)
    return np.array(vec,dtype=np.float32)

# UI — single place input
st.subheader("Enter Round for P1 (only one place)")
size_sel = st.selectbox("Size", ["S","B"])
color_sel = st.selectbox("Color", ["G","R","V","R+V","G+V"])
num_sel = st.number_input("Number", 0, 9, 0)

if st.button("Add Round"):
    entry = {"num":int(num_sel),"color":color_sel}
    st.session_state.history.append(entry)
    json.dump(st.session_state.history, open(HISTORY_FILE,"w"))
    st.success("Added round ✔️")

# TRAIN & PREDICT
if len(st.session_state.history) >= WINDOW:
    # Build window
    window_vec = [encode_round(x) for x in st.session_state.history[-WINDOW:]]
    X = np.stack(window_vec).reshape(1,WINDOW,FEATURES)

    # Predict number + color
    num_probs = st.session_state.num_model.predict(X, verbose=0)[0]
    col_probs = st.session_state.col_model.predict(X, verbose=0)[0]

    pred_num = int(np.argmax(num_probs))
    pred_col = INV_COLOR[int(np.argmax(col_probs))]
    pred_size = "B" if pred_num>=5 else "S"

    st.subheader("Prediction")
    st.write(f"Number → {pred_num} ({np.max(num_probs)*100:.1f}%)")
    st.write(f"Color → {pred_col} ({np.max(col_probs)*100:.1f}%)")
    st.write(f"Size → {pred_size}")

    # Train using replay
    if len(st.session_state.history) > WINDOW:
        prev_window = [encode_round(x) for x in st.session_state.history[-WINDOW-1:-1]]
        X_prev = np.stack(prev_window)
        Y_prev = {"num": st.session_state.history[-1]["num"],
                  "col": COLOR_MAP[st.session_state.history[-1]["color"]]}
        st.session_state.replay.add(X_prev, Y_prev)
        Xb, Yb = st.session_state.replay.sample(64)
        st.session_state.num_model.fit(Xb, Yb["num"], epochs=1, verbose=0)
        st.session_state.col_model.fit(Xb, Yb["col"], epochs=1, verbose=0)
        st.session_state.num_model.save(MODEL_NUM)
        st.session_state.col_model.save(MODEL_COL)

st.subheader("History")
st.dataframe(pd.DataFrame(st.session_state.history))
