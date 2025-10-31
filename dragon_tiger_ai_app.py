# app.py
"""
Dragon vs Tiger — Heavy LSTM + Ensemble Streamlit App
Requires: tensorflow, xgboost (optional), scikit-learn, streamlit, pandas, numpy, joblib
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import joblib
import random
from collections import defaultdict, Counter
from io import BytesIO

# Try to import heavy libs and provide graceful fallback messages
USE_XGBOOST = False
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier

USE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, GRU, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    USE_TF = True
except Exception:
    tf = None

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# safe rerun wrapper (supports streamlit versions)
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Page config
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger — LSTM Ensemble", layout="centered")
st.title("🐉 Dragon vs 🌟 Tiger — Heavy LSTM + Ensemble Predictor")

st.markdown("""
<style>
body { background-color: #0f1117; color: #ffffff; }
.stButton>button { background-color: #9c27b0; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session state defaults
# -------------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = []  # list of 'D','T','TIE'
if "X_train" not in st.session_state:
    st.session_state.X_train = []  # engineered features (numpy arrays)
if "y_train" not in st.session_state:
    st.session_state.y_train = []  # labels ints
if "seq_train" not in st.session_state:
    st.session_state.seq_train = []  # raw sequence arrays (timesteps,)
if "log" not in st.session_state:
    st.session_state.log = []
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))
if "sk_model" not in st.session_state:
    st.session_state.sk_model = None  # XGB / RF
if "tf_models" not in st.session_state:
    st.session_state.tf_models = {}  # keys: 'lstm', 'bilstm', 'gru'
if "last_trained" not in st.session_state:
    st.session_state.last_trained = None
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "onehot_enc" not in st.session_state:
    st.session_state.onehot_enc = None

# settings
if "context_len" not in st.session_state:
    st.session_state.context_len = 12  # default 10..20
if "min_train_examples" not in st.session_state:
    st.session_state.min_train_examples = 120  # heavy model needs more data
if "auto_retrain" not in st.session_state:
    st.session_state.auto_retrain = True
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False
if "ensemble_weights" not in st.session_state:
    # order: ['tf_ensemble', 'sk_model', 'markov', 'streak_bias']
    st.session_state.ensemble_weights = [0.5, 0.35, 0.1, 0.05]

# label maps
LABEL_MAP = {"D": 0, "T": 1, "TIE": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def encode_label(s):
    return LABEL_MAP.get(s, 2)

def decode_label(i):
    return INV_LABEL_MAP.get(int(i), "")

# -------------------------
# Feature builder utilities
# -------------------------
def build_engineered_features(history):
    """ Build engineered features from the last N history (N = context_len) """
    N = st.session_state.context_len
    lastN = history[-N:]
    if len(lastN) < N:
        lastN = ["TIE"] * (N - len(lastN)) + lastN
    enc = [LABEL_MAP.get(x, 2) for x in lastN]
    c = Counter(lastN)
    count_D = c.get("D", 0)
    count_T = c.get("T", 0)
    count_TIE = c.get("TIE", 0)
    last_winner = LABEL_MAP.get(lastN[-1], 2)
    # streak
    streak_len = 1
    for i in range(len(lastN)-2, -1, -1):
        if lastN[i] == lastN[-1]:
            streak_len += 1
        else:
            break
    prop_D = count_D / N
    prop_T = count_T / N
    # feature vector: encoded sequence (N), counts (3), last_winner, streak_len, prop_D, prop_T
    feats = np.array(enc + [count_D, count_T, count_TIE, last_winner, streak_len, prop_D, prop_T], dtype=float)
    return feats

def build_sequence_input(history):
    """ Build sequence array (timesteps, 1) from history for LSTM input """
    N = st.session_state.context_len
    lastN = history[-N:]
    if len(lastN) < N:
        lastN = ["TIE"] * (N - len(lastN)) + lastN
    seq = np.array([LABEL_MAP.get(x, 2) for x in lastN], dtype=float)
    # scale to 0..1
    seq = seq.reshape(-1, 1) / 2.0
    return seq  # shape (N,1)

# -------------------------
# Rebuild training sets from stored inputs
# -------------------------
def rebuild_training_from_inputs():
    st.session_state.X_train = []
    st.session_state.y_train = []
    st.session_state.seq_train = []
    inputs = st.session_state.inputs
    N = st.session_state.context_len
    for i in range(N, len(inputs)):
        hist = inputs[i-N:i]
        label = inputs[i]
        st.session_state.X_train.append(build_engineered_features(hist))
        st.session_state.y_train.append(encode_label(label))
        st.session_state.seq_train.append(build_sequence_input(hist))

# -------------------------
# Markov & streak helpers
# -------------------------
def markov_proba(last_seq):
    L = len(last_seq)
    for l in range(L, 4, -1):
        key = tuple(last_seq[-l:])
        counts = st.session_state.markov.get(key, {})
        if counts:
            total = sum(counts.values())
            return np.array([counts.get("D",0)/total, counts.get("T",0)/total, counts.get("TIE",0)/total])
    return None

def streak_bias_rule(last_seq):
    c = Counter(last_seq)
    N = len(last_seq)
    thr = max(3, int(0.6 * N))  # need at least 60% or at least 3
    if c.get("D",0) >= thr:
        return "T", 64
    if c.get("T",0) >= thr:
        return "D", 64
    return None, 0

# -------------------------
# Build & train sklearn tree model (XGBoost / RF)
# -------------------------
def train_sk_model(force=False):
    if len(st.session_state.X_train) < st.session_state.min_train_examples and not force:
        return False, f"Need at least {st.session_state.min_train_examples} engineered training examples to train SK model."
    X = np.vstack(st.session_state.X_train)
    y = np.array(st.session_state.y_train)
    weights = np.linspace(1.0, 3.0, len(y))
    if USE_XGBOOST:
        model = XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.85, use_label_encoder=False, eval_metric="mlogloss", verbosity=0, random_state=42)
        model.fit(X, y, sample_weight=weights)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced_subsample", random_state=42)
        model.fit(X, y)
    st.session_state.sk_model = model
    st.session_state.last_trained = time.time()
    return True, "SK model trained."

# -------------------------
# Build TF sequence model factory (returns compiled model)
# -------------------------
def build_seq_model(model_type="lstm", timesteps=None, features=1, lr=0.001):
    if not USE_TF:
        raise RuntimeError("TensorFlow not available.")
    if timesteps is None:
        timesteps = st.session_state.context_len
    inp = Input(shape=(timesteps, features), name="seq_in")
    # small additional features head (engineered features)
    eng_in = Input(shape=(st.session_state.context_len + 7,), name="eng_in")  # matches build_engineered_features length
    # sequence backbone
    x = inp
    if model_type == "lstm":
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32, return_sequences=False)(x)
    elif model_type == "bilstm":
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
    elif model_type == "gru":
        x = GRU(64, return_sequences=True)(x)
        x = GRU(32, return_sequences=False)(x)
    else:
        x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    # combine with engineered features
    combined = Concatenate()([x, eng_in])
    h = Dense(64, activation="relu")(combined)
    h = Dropout(0.2)(h)
    out = Dense(3, activation="softmax", name="out")(h)
    model = Model(inputs=[inp, eng_in], outputs=out)
    model.compile(optimizer=Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Train TF ensemble models (LSTM + BiLSTM + GRU)
# -------------------------
def train_tf_ensemble(force=False, epochs=20, batch_size=32):
    if not USE_TF:
        return False, "TensorFlow not installed."
    if len(st.session_state.seq_train) < st.session_state.min_train_examples and not force:
        return False, f"Need at least {st.session_state.min_train_examples} sequence examples for TF training."
    # prepare arrays
    X_seq = np.stack(st.session_state.seq_train)  # shape (samples, timesteps, 1)
    X_eng = np.vstack(st.session_state.X_train)   # engineered features shape (samples, eng_dim)
    y = np.array(st.session_state.y_train)
    # split
    Xs_train, Xs_val, Xe_train, Xe_val, y_train, y_val = train_test_split(X_seq, X_eng, y, test_size=0.12, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    timesteps = X_seq.shape[1]
    eng_dim = X_eng.shape[1]
    # build models
    models = {}
    for mtype in ["lstm", "bilstm", "gru"]:
        model = build_seq_model(mtype, timesteps=timesteps, features=1, lr=0.001)
        # fit with early stopping callback
        try:
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            model.fit({"seq_in": Xs_train, "eng_in": Xe_train}, y_train, validation_data=({"seq_in": Xs_val, "eng_in": Xe_val}, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
        except Exception as e:
            return False, f"TF training failed: {e}"
        models[mtype] = model
    # save into session
    st.session_state.tf_models = models
    st.session_state.last_trained = time.time()
    return True, "TF ensemble trained."

# -------------------------
# Predict using ensemble and heuristics
# -------------------------
def predict_now_heavy():
    inputs = st.session_state.inputs
    N = st.session_state.context_len
    if len(inputs) < N:
        return None, 0, "need_more"
    lastN = inputs[-N:]
    # engineered feature and seq
    eng_feat = build_engineered_features(lastN).reshape(1, -1)
    seq_feat = build_sequence_input(lastN).reshape(1, N, 1)
    # obtain model probs
    probs_tf = None
    if st.session_state.tf_models:
        # average TF ensemble probs
        probs = []
        for m in st.session_state.tf_models.values():
            p = m.predict({"seq_in": seq_feat, "eng_in": eng_feat}, verbose=0)[0]
            probs.append(p)
        probs_tf = np.mean(probs, axis=0)
    probs_sk = None
    if st.session_state.sk_model is not None:
        try:
            psk = st.session_state.sk_model.predict_proba(eng_feat)[0]
            probs_sk = np.array(psk)
        except Exception:
            probs_sk = None
    probs_markov = markov_proba(lastN)
    bias_pred, bias_conf = streak_bias_rule(lastN)

    # combine according to weights
    w_tf, w_sk, w_markov, w_bias = st.session_state.ensemble_weights
    # normalize weights in case model missing
    total_w = 0.0
    comps = []
    if probs_tf is not None:
        comps.append(("tf", probs_tf, w_tf)); total_w += w_tf
    if probs_sk is not None:
        comps.append(("sk", probs_sk, w_sk)); total_w += w_sk
    if probs_markov is not None:
        comps.append(("markov", probs_markov, w_markov)); total_w += w_markov

    if comps:
        combined = np.zeros(3, dtype=float)
        for name, p, w in comps:
            combined += p * (w / total_w)
        pred_idx = int(np.argmax(combined))
        pred_label = decode_label(pred_idx)
        conf = float(np.max(combined) * 100)
        # if confidence low, fall back to bias if applicable
        if conf < 60 and bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        return pred_label, round(conf), "ensemble"
    else:
        # no models available, fallback
        if bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        # fallback frequency
        counts = Counter(lastN)
        best = counts.most_common(1)[0][0]
        return best, 55, "frequency"

# -------------------------
# Persistence helpers
# -------------------------
def save_models_to_disk():
    os.makedirs("models", exist_ok=True)
    # save SK model
    if st.session_state.sk_model is not None:
        joblib.dump(st.session_state.sk_model, "models/sk_model.joblib")
    # save TF models
    if USE_TF and st.session_state.tf_models:
        for k, m in st.session_state.tf_models.items():
            m.save(f"models/tf_{k}.keras", include_optimizer=False)
    # save training data & metadata
    try:
        np.save("models/seq_train.npy", np.array(st.session_state.seq_train, dtype=object), allow_pickle=True)
        np.save("models/X_train.npy", np.array(st.session_state.X_train, dtype=object), allow_pickle=True)
        np.save("models/y_train.npy", np.array(st.session_state.y_train, dtype=object), allow_pickle=True)
        joblib.dump(dict(markov=st.session_state.markov, last_trained=st.session_state.last_trained), "models/meta.joblib")
    except Exception:
        pass
    return True

def load_models_from_disk():
    loaded = {"sk": False, "tf": False}
    if os.path.exists("models/sk_model.joblib"):
        try:
            st.session_state.sk_model = joblib.load("models/sk_model.joblib")
            loaded["sk"] = True
        except Exception:
            loaded["sk"] = False
    if USE_TF:
        tf_models = {}
        for k in ["lstm", "bilstm", "gru"]:
            p = f"models/tf_{k}.keras"
            if os.path.exists(p):
                try:
                    tf_models[k] = load_model(p, compile=False)
                except Exception:
                    tf_models[k] = None
        if tf_models:
            st.session_state.tf_models = {k: v for k, v in tf_models.items() if v is not None}
            loaded["tf"] = bool(st.session_state.tf_models)
    # try to load training arrays
    try:
        if os.path.exists("models/seq_train.npy"):
            st.session_state.seq_train = list(np.load("models/seq_train.npy", allow_pickle=True))
        if os.path.exists("models/X_train.npy"):
            st.session_state.X_train = list(np.load("models/X_train.npy", allow_pickle=True))
        if os.path.exists("models/y_train.npy"):
            st.session_state.y_train = list(np.load("models/y_train.npy", allow_pickle=True))
        if os.path.exists("models/meta.joblib"):
            meta = joblib.load("models/meta.joblib")
            st.session_state.markov = meta.get("markov", st.session_state.markov)
            st.session_state.last_trained = meta.get("last_trained", st.session_state.last_trained)
    except Exception:
        pass
    return loaded

# -------------------------
# UI Sidebar: settings & model controls
# -------------------------
with st.sidebar:
    st.header("Settings & Model Controls")
    st.subheader("Context window")
    ctx = st.slider("Context length (timesteps)", 10, 20, st.session_state.context_len)
    st.session_state.context_len = ctx

    st.subheader("Training")
    st.session_state.min_train_examples = st.number_input("Min train examples (TF)", min_value=30, max_value=2000, value=st.session_state.min_train_examples, step=10)
    st.session_state.auto_retrain = st.checkbox("Auto retrain when enough new data", value=st.session_state.auto_retrain)
    st.session_state.auto_mode = st.checkbox("Auto mode (predict next immediately after learning)", value=st.session_state.auto_mode)

    st.markdown("---")
    st.subheader("Ensemble weights (tf / sk / markov / bias)")
    w1 = st.number_input("TF weight", min_value=0.0, max_value=1.0, value=float(st.session_state.ensemble_weights[0]), step=0.05)
    w2 = st.number_input("SK weight", min_value=0.0, max_value=1.0, value=float(st.session_state.ensemble_weights[1]), step=0.05)
    w3 = st.number_input("Markov weight", min_value=0.0, max_value=1.0, value=float(st.session_state.ensemble_weights[2]), step=0.05)
    w4 = st.number_input("Bias weight", min_value=0.0, max_value=1.0, value=float(st.session_state.ensemble_weights[3]), step=0.01)
    total = w1 + w2 + w3 + w4
    if total == 0:
        st.warning("At least one ensemble weight should be > 0.")
    else:
        # normalize and set
        st.session_state.ensemble_weights = [w1/total, w2/total, w3/total, w4/total]

    st.markdown("---")
    st.subheader("Model actions")
    if st.button("Rebuild training from stored inputs"):
        rebuild_training_from_inputs()
        st.success("Rebuilt training arrays from inputs.")
    if st.button("Train SK model (XGB/RF) now"):
        ok, msg = train_sk_model(force=True)
        if ok: st.success(msg)
        else: st.warning(msg)
    if st.button("Train TF ensemble now (may be slow)"):
        ok, msg = train_tf_ensemble(force=True, epochs=30, batch_size=32)
        if ok: st.success(msg)
        else: st.warning(msg)
    if st.button("Save models to disk"):
        save_models_to_disk()
        st.success("Saved models & data to /models")
    if st.button("Load models from disk"):
        res = load_models_from_disk()
        st.info(f"Loaded (sk={res['sk']}, tf={res['tf']})")

    st.markdown("---")
    st.subheader("Info")
    st.text(f"TF installed: {'Yes' if USE_TF else 'No'}")
    st.text(f"XGBoost installed: {'Yes' if USE_XGBOOST else 'No'}")
    st.text(f"Training examples: {len(st.session_state.X_train)}")
    if st.session_state.last_trained:
        st.text("Last trained: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_trained)))
    else:
        st.text("Last trained: N/A")

# -------------------------
# Main UI: Enter result (Add) and dev tools
# -------------------------
st.subheader("🎮 Add Result (D / T / TIE)")
col1, col2 = st.columns([3,1])
with col1:
    choice = st.selectbox("Choose Result", ["D","T","TIE"], key="choice_select")
with col2:
    if st.button("Add Result"):
        st.session_state.inputs.append(choice)
        st.success(f"Added: {choice}")
        # append training rows if possible
        i = len(st.session_state.inputs) - 1
        N = st.session_state.context_len
        if i >= N:
            hist = st.session_state.inputs[i-N:i]
            st.session_state.X_train.append(build_engineered_features(hist))
            st.session_state.y_train.append(encode_label(st.session_state.inputs[i]))
            st.session_state.seq_train.append(build_sequence_input(hist))
            # update markov
            for l in range(N,4,-1):
                if i >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][st.session_state.inputs[i]] += 1
        # optionally auto retrain SK or TF
        if st.session_state.auto_retrain and len(st.session_state.X_train) >= st.session_state.min_train_examples:
            # train SK quick model on engineered features (fast)
            _ok, _msg = train_sk_model(force=False)
            # Do not automatically run heavy TF training unless user explicitly asks (because it's slow)
        if st.session_state.auto_mode:
            safe_rerun()

with st.expander("Developer tools"):
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Add 20 random rounds"):
            for _ in range(20):
                st.session_state.inputs.append(random.choice(["D","T","TIE"]))
            rebuild_training_from_inputs()
            st.success("Added 20 random rounds.")
    with c2:
        if st.button("Clear all data"):
            st.session_state.inputs = []
            st.session_state.X_train = []
            st.session_state.y_train = []
            st.session_state.seq_train = []
            st.session_state.log = []
            st.session_state.markov = defaultdict(lambda: defaultdict(int))
            st.session_state.sk_model = None
            st.session_state.tf_models = {}
            st.success("Cleared everything.")
    with c3:
        if st.button("Force retrain TF (30 epoches)"):
            ok, msg = train_tf_ensemble(force=True, epochs=30, batch_size=32)
            if ok: st.success(msg)
            else: st.warning(msg)

# -------------------------
# Prediction panel (heavy ensemble)
# -------------------------
st.markdown("---")
st.subheader("🔮 Predict Next Round (LSTM Ensemble + Heuristics)")
if len(st.session_state.inputs) < st.session_state.context_len:
    st.info(f"Provide {st.session_state.context_len - len(st.session_state.inputs)} more rounds to enable predictions.")
else:
    pred, conf, source = predict_now_heavy()
    st.text(f"Source: {source}   (weights: tf={st.session_state.ensemble_weights[0]:.2f}, sk={st.session_state.ensemble_weights[1]:.2f}, markov={st.session_state.ensemble_weights[2]:.2f}, bias={st.session_state.ensemble_weights[3]:.2f})")
    st.text(f"Training examples: {len(st.session_state.X_train)}")
    if pred is None:
        st.warning("Prediction unavailable.")
    else:
        if source == "ensemble" and conf >= 60:
            st.success(f"Prediction: **{pred}** | Confidence: {conf}%  (ensemble)")
            st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        elif source == "streak-bias":
            st.warning(f"Prediction (streak-bias): **{pred}** | Confidence: {conf}%")
            st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
        else:
            st.info(f"Prediction (fallback): **{pred}** | Confidence: {conf}%")

        # training balance
        labels_preview = [decode_label(x) for x in st.session_state.y_train]
        st.text(f"Training Balance ➡️ D: {labels_preview.count('D')} | T: {labels_preview.count('T')} | TIE: {labels_preview.count('TIE')}")
        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions in row. Be careful.")

        # Confirm & Learn: user provides actual outcome -> model learns and optionally retrains
        actual = st.selectbox("Enter actual result (to teach the model)", ["D","T","TIE"], key="confirm_actual")
        if st.button("Confirm & Learn"):
            correct = (actual == pred)
            st.session_state.log.append({
                "Prediction": pred,
                "Confidence": conf,
                "Source": source,
                "Actual": actual,
                "Correct": "✅" if correct else "❌",
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            # add training example (before appending actual)
            hist = st.session_state.inputs[-st.session_state.context_len:]
            st.session_state.X_train.append(build_engineered_features(hist))
            st.session_state.y_train.append(encode_label(actual))
            st.session_state.seq_train.append(build_sequence_input(hist))
            # update markov
            i = len(st.session_state.inputs)
            for l in range(st.session_state.context_len, 4, -1):
                if i >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][actual] += 1
            # append actual to history
            st.session_state.inputs.append(actual)
            if correct:
                st.session_state.loss_streak = 0
            else:
                st.session_state.loss_streak += 1
            # auto retrain SK (fast) if enough data
            if st.session_state.auto_retrain and len(st.session_state.X_train) >= 50:
                train_sk_model(force=False)
            # do NOT auto-run heavy TF training by default (slow). If user enabled auto_retrain AND wants TF, they can click button in sidebar
            if st.session_state.auto_mode:
                safe_rerun()

# -------------------------
# History & Exports
# -------------------------
st.markdown("---")
st.subheader("📈 History & Exports")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export History to Excel"):
            buf = BytesIO()
            df.to_excel(buf, index=False)
            st.download_button("⬇️ Download Excel", data=buf.getvalue(), file_name="prediction_history.xlsx")
    with c2:
        if st.button("Export Training Set (CSV)"):
            if st.session_state.X_train:
                X = np.vstack(st.session_state.X_train)
                y = np.array(st.session_state.y_train)
                cols = [f"enc_{i+1}" for i in range(st.session_state.context_len)] + ["count_D","count_T","count_TIE","last_winner","streak_len","prop_D","prop_T"]
                tr_df = pd.DataFrame(X, columns=cols)
                tr_df["label"] = [decode_label(int(x)) for x in y]
                buf = BytesIO()
                tr_df.to_csv(buf, index=False)
                st.download_button("⬇️ Download training CSV", data=buf.getvalue(), file_name="training_set.csv")
            else:
                st.warning("No training data to export.")
else:
    st.info("No history yet. Use 'Confirm & Learn' after predictions to log data.")

st.markdown("---")
st.caption("Built with ❤️ — LSTM/TF ensemble + XGBoost/RF + Markov/streak heuristics. Heavy models require adequate examples and compute.")
