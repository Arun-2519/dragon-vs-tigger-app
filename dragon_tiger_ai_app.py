#!/usr/bin/env python3
"""
telegram_dragon_tiger_bot.py

Hybrid Dragon-Tiger predictor with Telegram bot interface,
hybrid models (XGBoost + LSTM ensemble), Markov/streak pattern logic,
auto daily retrain, continuous learning from user-confirmed results.

Configuration: set TELEGRAM_TOKEN (env var) or edit TELEGRAM_TOKEN below.
Place a CSV file with historical results (column name 'result') next to this script.

Author: Generated (adapt before production)
"""

import os
import time
import threading
import traceback
import joblib
from datetime import datetime
from functools import wraps
from collections import defaultdict, Counter
from io import BytesIO

# Machine learning / DL
import numpy as np
import pandas as pd

# XGBoost / sklearn
try:
    from xgboost import XGBClassifier, DMatrix, Booster
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGBOOST = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, GRU, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    USE_TF = True
except Exception:
    USE_TF = False

# Telegram bot
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Scheduler for daily retrain
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------------------
# CONFIG (edit / override via env vars)
# ---------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7899395894:AAFD7G9y9qnj7QiCfrjrXi8JMcbO1q4Q33M")
CSV_PATH = os.environ.get("CSV_PATH", "D_vs_T_results.csv")
CSV_COLUMN = os.environ.get("CSV_COLUMN", "result")  # column name inside CSV
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", 12))  # 10..20 recommended
AUTO_RETRAIN_HOUR = int(os.environ.get("AUTO_RETRAIN_HOUR", 3))  # daily retrain hour (0-23)
MIN_SK_TRAIN = int(os.environ.get("MIN_SK_TRAIN", 30))
MIN_TF_TRAIN = int(os.environ.get("MIN_TF_TRAIN", 80))

# Create models dir
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
LABEL_MAP = {"D": 0, "T": 1, "TIE": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def encode_label(s):
    return LABEL_MAP.get(s.upper(), 2)

def decode_label(i):
    return INV_LABEL_MAP.get(int(i), "TIE")

def safe_print(*args, **kwargs):
    print(datetime.utcnow().isoformat(), *args, **kwargs)

# decorator for handlers to catch exceptions
def handler_safe(func):
    @wraps(func)
    def wrapper(update: Update, context: CallbackContext):
        try:
            return func(update, context)
        except Exception as e:
            safe_print("Handler error:", e)
            safe_print(traceback.format_exc())
            update.message.reply_text("⚠️ An internal error occurred. Check logs.")
    return wrapper

# ---------------------------
# Data and Pattern Store
# ---------------------------
class DataStore:
    def __init__(self):
        self.history = []          # list of 'D','T','TIE' (strings)
        self.X_eng = []            # list of engineered features
        self.y = []                # labels (ints)
        self.seq = []              # sequence arrays (timesteps,1)
        self.markov = defaultdict(lambda: defaultdict(int))
        # models
        self.sk_model = None       # XGBoost or RandomForest
        self.tf_models = {}        # dict: 'lstm','bilstm','gru'
        # recent performance
        self.recent_accuracy = {"sk": 0.5, "tf": 0.5}
        # meta
        self.last_trained = None
        self.lock = threading.RLock()

    def save(self):
        # persist models & arrays
        try:
            joblib.dump(self.history, os.path.join(MODELS_DIR, "history.pkl"))
            joblib.dump(self.X_eng, os.path.join(MODELS_DIR, "X_eng.pkl"))
            joblib.dump(self.y, os.path.join(MODELS_DIR, "y.pkl"))
            joblib.dump(dict(self.markov), os.path.join(MODELS_DIR, "markov.pkl"))
            if self.sk_model is not None:
                if USE_XGBOOST and hasattr(self.sk_model, "save_model"):
                    self.sk_model.save_model(os.path.join(MODELS_DIR, "sk_model.xgb"))
                else:
                    joblib.dump(self.sk_model, os.path.join(MODELS_DIR, "sk_model.joblib"))
            if USE_TF and self.tf_models:
                for k, m in self.tf_models.items():
                    try:
                        m.save(os.path.join(MODELS_DIR, f"tf_{k}.keras"), include_optimizer=False)
                    except Exception:
                        safe_print("Failed to save TF model", k)
            safe_print("Saved models & data.")
        except Exception as e:
            safe_print("Save failed:", e)

    def load(self):
        try:
            if os.path.exists(os.path.join(MODELS_DIR, "history.pkl")):
                self.history = joblib.load(os.path.join(MODELS_DIR, "history.pkl"))
            if os.path.exists(os.path.join(MODELS_DIR, "X_eng.pkl")):
                self.X_eng = joblib.load(os.path.join(MODELS_DIR, "X_eng.pkl"))
            if os.path.exists(os.path.join(MODELS_DIR, "y.pkl")):
                self.y = joblib.load(os.path.join(MODELS_DIR, "y.pkl"))
            if os.path.exists(os.path.join(MODELS_DIR, "markov.pkl")):
                m = joblib.load(os.path.join(MODELS_DIR, "markov.pkl"))
                self.markov = defaultdict(lambda: defaultdict(int), m)
            if USE_XGBOOST and os.path.exists(os.path.join(MODELS_DIR, "sk_model.xgb")):
                b = Booster()
                b.load_model(os.path.join(MODELS_DIR, "sk_model.xgb"))
                self.sk_model = b
            elif os.path.exists(os.path.join(MODELS_DIR, "sk_model.joblib")):
                self.sk_model = joblib.load(os.path.join(MODELS_DIR, "sk_model.joblib"))
            if USE_TF:
                for name in ["lstm", "bilstm", "gru"]:
                    p = os.path.join(MODELS_DIR, f"tf_{name}.keras")
                    if os.path.exists(p):
                        try:
                            self.tf_models[name] = load_model(p, compile=False)
                        except Exception:
                            safe_print("Failed load TF model:", name)
            safe_print("Loaded models & data (if present).")
        except Exception as e:
            safe_print("Load error:", e)

DATA = DataStore()

# ---------------------------
# Feature builders & models
# ---------------------------
def build_engineered_features(history, context_len=CONTEXT_LEN):
    N = context_len
    lastN = history[-N:]
    if len(lastN) < N:
        lastN = ["TIE"]*(N - len(lastN)) + lastN
    enc = [LABEL_MAP.get(x, 2) for x in lastN]
    c = Counter(lastN)
    count_D = c.get('D', 0); count_T = c.get('T', 0); count_TIE = c.get('TIE', 0)
    last_winner = LABEL_MAP.get(lastN[-1], 2)
    streak_len = 1
    for i in range(len(lastN)-2, -1, -1):
        if lastN[i] == lastN[-1]:
            streak_len += 1
        else:
            break
    prop_D = count_D / N
    prop_T = count_T / N
    features = np.array(enc + [count_D, count_T, count_TIE, last_winner, streak_len, prop_D, prop_T], dtype=float)
    return features

def build_sequence_input(history, context_len=CONTEXT_LEN):
    N = context_len
    lastN = history[-N:]
    if len(lastN) < N:
        lastN = ["TIE"]*(N - len(lastN)) + lastN
    seq = np.array([LABEL_MAP.get(x, 2) for x in lastN], dtype=float)
    seq = seq.reshape(N, 1) / 2.0
    return seq

# TF model factory
def build_seq_model(model_type, timesteps=CONTEXT_LEN, eng_dim=None, lr=0.001):
    if not USE_TF:
        raise RuntimeError("TensorFlow not available")
    if eng_dim is None:
        raise ValueError("eng_dim required")
    inp = Input(shape=(timesteps, 1), name="seq_in")
    eng_in = Input(shape=(eng_dim,), name="eng_in")
    if model_type == "lstm":
        x = LSTM(64, return_sequences=True)(inp)
        x = LSTM(32, return_sequences=False)(x)
    elif model_type == "bilstm":
        x = Bidirectional(LSTM(64, return_sequences=True))(inp)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
    elif model_type == "gru":
        x = GRU(64, return_sequences=True)(inp)
        x = GRU(32, return_sequences=False)(x)
    else:
        x = LSTM(64, return_sequences=False)(inp)
    x = Dropout(0.2)(x)
    combined = Concatenate()([x, eng_in])
    h = Dense(64, activation="relu")(combined)
    h = Dropout(0.2)(h)
    out = Dense(3, activation="softmax")(h)
    model = Model([inp, eng_in], out)
    model.compile(optimizer=Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# Training Routines
# ---------------------------
def rebuild_training_from_history(context_len=CONTEXT_LEN):
    with DATA.lock:
        X_eng = []
        seqs = []
        y = []
        for i in range(context_len, len(DATA.history)):
            hist_slice = DATA.history[i-context_len:i]
            X_eng.append(build_engineered_features(hist_slice, context_len))
            seqs.append(build_sequence_input(hist_slice, context_len))
            y.append(encode_label(DATA.history[i]))
        if X_eng:
            return np.vstack(X_eng), np.stack(seqs), np.array(y)
        return None, None, None

def train_sk_model(force=False):
    """Train XGBoost (or RandomForest fallback) on engineered features."""
    X_eng, seqs, y = rebuild_training_from_history(CONTEXT_LEN)
    if X_eng is None:
        return False, "No training data"
    if len(X_eng) < MIN_SK_TRAIN and not force:
        return False, f"Need >= {MIN_SK_TRAIN} samples for SK training"
    try:
        if USE_XGBOOST:
            model = XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.85, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
            model.fit(X_eng, y)
        else:
            model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced_subsample", random_state=42)
            model.fit(X_eng, y)
        with DATA.lock:
            DATA.sk_model = model
        # quick holdout accuracy
        try:
            Xtr, Xv, ytr, yv = train_test_split(X_eng, y, test_size=0.12, random_state=42, stratify=y if len(np.unique(y))>1 else None)
            if USE_XGBOOST:
                preds = np.argmax(model.predict_proba(Xv), axis=1) if hasattr(model, "predict_proba") else np.argmax(model.predict(DMatrix(Xv)), axis=1)
            else:
                preds = model.predict(Xv)
            acc = float(np.mean(preds == yv))
            DATA.recent_accuracy["sk"] = acc
        except Exception:
            pass
        DATA.last_trained = time.time()
        return True, "SK model trained"
    except Exception as e:
        safe_print("SK train error:", e)
        return False, str(e)

def train_tf_ensemble(force=False, epochs=15, batch_size=32):
    """Train TF ensemble (LSTM/BiLSTM/GRU). This is heavier and may take time."""
    if not USE_TF:
        return False, "TensorFlow not installed"
    X_eng, seqs, y = rebuild_training_from_history(CONTEXT_LEN)
    if seqs is None:
        return False, "No training data"
    if len(seqs) < MIN_TF_TRAIN and not force:
        return False, f"Need >= {MIN_TF_TRAIN} sequence samples for TF"
    try:
        timesteps = seqs.shape[1]
        eng_dim = X_eng.shape[1]
        Xs_train, Xs_val, Xe_train, Xe_val, y_train, y_val = train_test_split(seqs, X_eng, y, test_size=0.12, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        models = {}
        for mtype in ["lstm", "bilstm", "gru"]:
            m = build_seq_model(mtype, timesteps, eng_dim)
            es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            m.fit({"seq_in": Xs_train, "eng_in": Xe_train}, y_train, validation_data=({"seq_in": Xs_val, "eng_in": Xe_val}, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
            # measure val acc
            try:
                p = m.predict({"seq_in": Xs_val, "eng_in": Xe_val}, verbose=0)
                acc = float(np.mean(np.argmax(p, axis=1) == y_val))
                DATA.recent_accuracy["tf"] = max(DATA.recent_accuracy.get("tf", 0.5), acc)
            except Exception:
                pass
            models[mtype] = m
        with DATA.lock:
            DATA.tf_models = models
        DATA.last_trained = time.time()
        return True, "TF ensemble trained"
    except Exception as e:
        safe_print("TF train error:", e)
        return False, str(e)

# ---------------------------
# Prediction: ensemble + heuristics
# ---------------------------
def markov_proba(last_seq):
    for l in range(len(last_seq), 4, -1):
        key = tuple(last_seq[-l:])
        counts = DATA.markov.get(key, {})
        if counts:
            total = sum(counts.values())
            return np.array([counts.get("D",0)/total, counts.get("T",0)/total, counts.get("TIE",0)/total])
    return None

def streak_bias_rule(last_seq):
    c = Counter(last_seq)
    N = len(last_seq)
    thr = max(3, int(0.6 * N))
    if c.get("D", 0) >= thr:
        return "T", 64
    if c.get("T", 0) >= thr:
        return "D", 64
    return None, 0

def ensemble_predict(context_len=CONTEXT_LEN, weights=None):
    with DATA.lock:
        hist = list(DATA.history)
        sk_model = DATA.sk_model
        tf_models = dict(DATA.tf_models)
    if len(hist) < context_len:
        return None, 0, "need_more"
    lastN = hist[-context_len:]
    eng_feat = build_engineered_features(lastN, context_len).reshape(1, -1)
    seq_feat = build_sequence_input(lastN, context_len).reshape(1, context_len, 1)
    probs_tf = None
    if tf_models:
        arr = []
        for m in tf_models.values():
            try:
                p = m.predict({"seq_in": seq_feat, "eng_in": eng_feat}, verbose=0)[0]
                arr.append(p)
            except Exception:
                pass
        if arr:
            probs_tf = np.mean(arr, axis=0)
    probs_sk = None
    if sk_model is not None:
        try:
            if USE_XGBOOST and isinstance(sk_model, Booster):
                dm = DMatrix(eng_feat)
                probs_sk = sk_model.predict(dm)[0]
            elif USE_XGBOOST:
                probs_sk = sk_model.predict_proba(eng_feat)[0]
            else:
                probs_sk = sk_model.predict_proba(eng_feat)[0]
            probs_sk = np.array(probs_sk)
        except Exception:
            probs_sk = None
    probs_markov = markov_proba(lastN)
    bias_pred, bias_conf = streak_bias_rule(lastN)
    # default dynamic weights
    if weights is None:
        w_tf = 0.5 if probs_tf is not None else 0.0
        w_sk = 0.35 if probs_sk is not None else 0.0
        w_mark = 0.15 if probs_markov is not None else 0.0
        total = w_tf + w_sk + w_mark
        if total == 0:
            if bias_pred:
                return bias_pred, bias_conf, "streak-bias"
            cnt = Counter(lastN)
            return cnt.most_common(1)[0][0], 55, "frequency"
        w_tf /= total; w_sk /= total; w_mark /= total
    else:
        w_tf, w_sk, w_mark = weights
    comps = []
    total_w = 0.0
    if probs_tf is not None:
        comps.append((probs_tf, w_tf)); total_w += w_tf
    if probs_sk is not None:
        comps.append((probs_sk, w_sk)); total_w += w_sk
    if probs_markov is not None:
        comps.append((probs_markov, w_mark)); total_w += w_mark
    if comps:
        combined = np.zeros(3)
        for p, w in comps:
            combined += p * (w/total_w)
        idx = int(np.argmax(combined))
        conf = float(np.max(combined) * 100)
        if conf < 60 and bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        return decode_label(idx), round(conf), "ensemble"
    else:
        if bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        cnt = Counter(lastN)
        return cnt.most_common(1)[0][0], 55, "frequency"

# ---------------------------
# Incremental online update
# ---------------------------
def online_learn(new_actual):
    """
    Called when the user confirms an actual outcome.
    Adds to history and does a light online update:
      - append engineered features and seq to training arrays
      - update markov counts
      - do one LSTM epoch on the single new sample (fast)
      - retrain SK (fast) if enough new data (background thread)
    """
    with DATA.lock:
        i = len(DATA.history)
        # append training sample using last CONTEXT_LEN history (before appending actual)
        if i >= CONTEXT_LEN:
            hist_slice = DATA.history[i-CONTEXT_LEN:i]
            DATA.X_eng.append(build_engineered_features(hist_slice, CONTEXT_LEN))
            DATA.seq.append(build_sequence_input(hist_slice, CONTEXT_LEN))
            DATA.y.append(encode_label(new_actual))
            for l in range(CONTEXT_LEN, 4, -1):
                if i >= l:
                    key = tuple(DATA.history[i-l:i])
                    DATA.markov[key][new_actual] += 1
        DATA.history.append(new_actual)

    # light TF update: fit one epoch on the new sample if TF models exist
    if USE_TF and DATA.tf_models:
        try:
            # prepare single-sample arrays
            Xs = np.stack(DATA.seq[-1:])  # shape (1, timesteps, 1)
            Xe = np.vstack(DATA.X_eng[-1:])
            y = np.array([DATA.y[-1]])
            for m in DATA.tf_models.values():
                try:
                    m.fit({"seq_in": Xs, "eng_in": Xe}, y, epochs=1, batch_size=1, verbose=0)
                except Exception:
                    pass
        except Exception:
            pass

    # schedule SK retrain in background if enough data
    def retrain_sk_bg():
        ok, msg = train_sk_model(force=False)
        safe_print("Auto SK retrain:", ok, msg)
    # Fire background retrain if meeting conditions
    if len(DATA.X_eng) >= MIN_SK_TRAIN:
        t = threading.Thread(target=retrain_sk_bg, daemon=True)
        t.start()

# ---------------------------
# CSV load & initial training
# ---------------------------
def load_csv_and_prepare(path=CSV_PATH, csv_column=CSV_COLUMN):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if csv_column not in df.columns:
        raise ValueError(f"CSV missing column '{csv_column}'")
    # sanitize values to D/T/TIE uppercase
    values = df[csv_column].astype(str).str.strip().str.upper().tolist()
    values = [v if v in LABEL_MAP else "TIE" for v in values]
    with DATA.lock:
        DATA.history = values.copy()
        # reset training arrays and build from history
        DATA.X_eng = []; DATA.seq = []; DATA.y = []
        for i in range(CONTEXT_LEN, len(DATA.history)):
            hist_slice = DATA.history[i-CONTEXT_LEN:i]
            DATA.X_eng.append(build_engineered_features(hist_slice, CONTEXT_LEN))
            DATA.seq.append(build_sequence_input(hist_slice, CONTEXT_LEN))
            DATA.y.append(encode_label(DATA.history[i]))
            # markov update
            for l in range(CONTEXT_LEN, 4, -1):
                if i >= l:
                    key = tuple(DATA.history[i-l:i])
                    DATA.markov[key][DATA.history[i]] += 1
    safe_print(f"Loaded CSV with {len(DATA.history)} rounds; training examples: {len(DATA.X_eng)}")
    return True

# ---------------------------
# Scheduler (auto daily retrain)
# ---------------------------
scheduler = BackgroundScheduler()
def daily_retrain_job():
    safe_print("Daily retrain job started")
    try:
        ok, msg = train_sk_model(force=False)
        safe_print("SK retrain:", ok, msg)
        # TF retrain only if enough data
        X_eng, seqs, y = rebuild_training_from_history(CONTEXT_LEN)
        if USE_TF and seqs is not None and len(seqs) >= MIN_TF_TRAIN:
            ok2, msg2 = train_tf_ensemble(force=False, epochs=20, batch_size=32)
            safe_print("TF retrain:", ok2, msg2)
    except Exception as e:
        safe_print("Daily retrain error:", e)

# schedule daily at configured hour
scheduler.add_job(daily_retrain_job, 'cron', hour=AUTO_RETRAIN_HOUR)
scheduler.start()

# ---------------------------
# Telegram bot handlers
# ---------------------------

@handler_safe
def start_handler(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🐉 Dragon-Tiger Bot\n"
        "Commands:\n"
        "/predict - predict next using recent history\n"
        "/add <D/T/TIE> - add recent result (does NOT teach model until /confirm)\n"
        "/confirm <D/T/TIE> - confirm actual result and teach model (online learning)\n"
        "/history - show last 30 rounds\n"
        "/status - show model status & training counts\n"
        "/retrain - force retrain SK and (optionally) TF\n"
        "/save - save models & data\n"
        "/load - load models & data\n"
    )

@handler_safe
def predict_handler(update: Update, context: CallbackContext):
    pred, conf, src = ensemble_predict(CONTEXT_LEN)
    if pred is None:
        update.message.reply_text(f"Need {CONTEXT_LEN - len(DATA.history)} more rounds to predict.")
        return
    update.message.reply_text(f"🔮 Prediction: *{pred}*  Confidence: *{conf}%*  (source: {src})", parse_mode=ParseMode.MARKDOWN)

@handler_safe
def add_handler(update: Update, context: CallbackContext):
    # /add D
    args = context.args
    if not args:
        update.message.reply_text("Usage: /add D  or /add T  or /add TIE")
        return
    val = args[0].upper()
    if val not in LABEL_MAP:
        update.message.reply_text("Value must be one of D / T / TIE")
        return
    with DATA.lock:
        DATA.history.append(val)
        i = len(DATA.history) - 1
        if i >= CONTEXT_LEN:
            hist_slice = DATA.history[i-CONTEXT_LEN:i]
            DATA.X_eng.append(build_engineered_features(hist_slice, CONTEXT_LEN))
            DATA.seq.append(build_sequence_input(hist_slice, CONTEXT_LEN))
            DATA.y.append(encode_label(DATA.history[i]))
            for l in range(CONTEXT_LEN, 4, -1):
                if i >= l:
                    key = tuple(DATA.history[i-l:i])
                    DATA.markov[key][DATA.history[i]] += 1
    update.message.reply_text(f"Added round: {val}. Current rounds: {len(DATA.history)}")
    # optionally auto-retrain SK
    if len(DATA.X_eng) >= MIN_SK_TRAIN:
        t = threading.Thread(target=train_sk_model, kwargs={"force": False}, daemon=True)
        t.start()

@handler_safe
def confirm_handler(update: Update, context: CallbackContext):
    # /confirm D
    args = context.args
    if not args:
        update.message.reply_text("Usage: /confirm D  or /confirm T  or /confirm TIE")
        return
    val = args[0].upper()
    if val not in LABEL_MAP:
        update.message.reply_text("Value must be one of D / T / TIE")
        return
    online_learn(val)
    update.message.reply_text(f"Confirmed and learned: {val}. Total rounds: {len(DATA.history)}")

@handler_safe
def history_handler(update: Update, context: CallbackContext):
    with DATA.lock:
        last = DATA.history[-50:]
    if not last:
        update.message.reply_text("No history yet.")
        return
    # display as simple string
    s = " ".join(last[-30:])
    update.message.reply_text(f"Last rounds (most recent last):\n{ s }")

@handler_safe
def status_handler(update: Update, context: CallbackContext):
    with DATA.lock:
        n_rounds = len(DATA.history)
        n_train = len(DATA.X_eng)
        sk = DATA.sk_model is not None
        tf = bool(DATA.tf_models)
    last_trained = datetime.utcfromtimestamp(DATA.last_trained).isoformat() if DATA.last_trained else "N/A"
    msg = (
        f"Rounds: {n_rounds}\n"
        f"Training examples: {n_train}\n"
        f"SK model: {'yes' if sk else 'no'}\n"
        f"TF models: {'yes' if tf else 'no'}\n"
        f"Recent acc (sk/tf): {DATA.recent_accuracy.get('sk',0):.2f} / {DATA.recent_accuracy.get('tf',0):.2f}\n"
        f"Last trained: {last_trained}\n"
        f"Context length: {CONTEXT_LEN}"
    )
    update.message.reply_text(msg)

@handler_safe
def retrain_handler(update: Update, context: CallbackContext):
    update.message.reply_text("Starting retrain (SK fast, TF optional). This runs in background.")
    def bg():
        ok, msg = train_sk_model(force=True)
        safe_print("Manual SK retrain:", ok, msg)
        update.message.reply_text(f"SK retrain: {msg}")
        # if user asked for TF in args or we have enough data, train TF
        if ("tf" in context.args) or (USE_TF and len(DATA.seq) >= MIN_TF_TRAIN):
            ok2, msg2 = train_tf_ensemble(force=True, epochs=20, batch_size=32)
            safe_print("Manual TF retrain:", ok2, msg2)
            update.message.reply_text(f"TF retrain: {msg2}")
    threading.Thread(target=bg, daemon=True).start()

@handler_safe
def save_handler(update: Update, context: CallbackContext):
    DATA.save()
    update.message.reply_text("Saved models & data to models/")

@handler_safe
def load_handler(update: Update, context: CallbackContext):
    DATA.load()
    update.message.reply_text("Loaded models & data (if present).")

# ---------------------------
# Main bot start
# ---------------------------
def main():
    # Load CSV data (if exists)
    try:
        if os.path.exists(CSV_PATH):
            load_csv_and_prepare(CSV_PATH, CSV_COLUMN)
    except Exception as e:
        safe_print("CSV load error:", e)

    # Start Telegram bot
    if TELEGRAM_TOKEN is None or TELEGRAM_TOKEN.strip() == "" or TELEGRAM_TOKEN.startswith("<PUT"):
        safe_print("ERROR: Set TELEGRAM_TOKEN env var or edit script.")
        return
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CommandHandler("predict", predict_handler))
    dp.add_handler(CommandHandler("add", add_handler))
    dp.add_handler(CommandHandler("confirm", confirm_handler))
    dp.add_handler(CommandHandler("history", history_handler))
    dp.add_handler(CommandHandler("status", status_handler))
    dp.add_handler(CommandHandler("retrain", retrain_handler))
    dp.add_handler(CommandHandler("save", save_handler))
    dp.add_handler(CommandHandler("load", load_handler))
    # fallback message handler
    def echo(update, context):
        update.message.reply_text("Use /predict, /add <D>, /confirm <D>, /history, /status, /retrain, /save, /load")
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    safe_print("Starting Telegram bot...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
