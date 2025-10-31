# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import os
import time
from collections import defaultdict, Counter
from io import BytesIO

# Model selection (xgboost preferred)
USE_XGBOOST = False
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier

# safe rerun wrapper (handles new/old streamlit)
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Page setup
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger Predictor (Updated)", layout="centered")
st.title("🐉 Dragon vs 🌟 Tiger Predictor — Continuous Learning")

st.markdown("""
    <style>
        body { background-color: #0f1117; color: #ffffff; }
        .stButton>button { background-color: #9c27b0; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------
# Session state defaults
# ---------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = []        # history of rounds: 'D','T','TIE'
if "X_train" not in st.session_state:
    st.session_state.X_train = []       # feature vectors
if "y_train" not in st.session_state:
    st.session_state.y_train = []       # labels 0/1/2
if "log" not in st.session_state:
    st.session_state.log = []           # prediction history
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))
if "model" not in st.session_state:
    st.session_state.model = None
if "last_trained" not in st.session_state:
    st.session_state.last_trained = None
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0

# Settings: context length, thresholds, auto-train toggle
if "context_len" not in st.session_state:
    st.session_state.context_len = 10   # default 10; user can set 10..20
if "min_train_examples" not in st.session_state:
    st.session_state.min_train_examples = 40
if "auto_retrain" not in st.session_state:
    st.session_state.auto_retrain = True
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False

# label maps
LABEL_MAP = {"D": 0, "T": 1, "TIE": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def encode_label(s):
    return LABEL_MAP.get(s, 2)

def decode_label(i):
    return INV_LABEL_MAP.get(int(i), "")

# ---------------------
# Feature builder (uses context length)
# ---------------------
def build_features(history):
    """
    history: list of last context_len outcomes (strings).
    returns feature vector:
      - encoded last N (N = context_len)
      - counts: count_D, count_T, count_TIE
      - last_winner (int)
      - last_streak_len (int)
      - prop_D, prop_T
    """
    N = st.session_state.context_len
    lastN = history[-N:]
    # pad if short (shouldn't usually happen when used for training)
    if len(lastN) < N:
        pad = ["TIE"] * (N - len(lastN))
        lastN = pad + lastN
    encoded = [LABEL_MAP.get(x, 2) for x in lastN]
    c = Counter(lastN)
    count_D = c.get("D", 0)
    count_T = c.get("T", 0)
    count_TIE = c.get("TIE", 0)
    last_winner = LABEL_MAP.get(lastN[-1], 2)
    # compute streak length (ending streak)
    streak_len = 1
    for i in range(len(lastN)-2, -1, -1):
        if lastN[i] == lastN[-1]:
            streak_len += 1
        else:
            break
    prop_D = count_D / N
    prop_T = count_T / N
    features = encoded + [count_D, count_T, count_TIE, last_winner, streak_len, prop_D, prop_T]
    return np.array(features, dtype=float)

# ---------------------
# Add training examples from inputs (construct X_train/y_train)
# ---------------------
def rebuild_training_from_inputs():
    st.session_state.X_train = []
    st.session_state.y_train = []
    inputs = st.session_state.inputs
    N = st.session_state.context_len
    for i in range(N, len(inputs)):
        hist = inputs[i-N:i]
        label = inputs[i]
        st.session_state.X_train.append(build_features(hist))
        st.session_state.y_train.append(encode_label(label))

# ---------------------
# Training function
# ---------------------
def train_model(force=False):
    if len(st.session_state.X_train) < st.session_state.min_train_examples and not force:
        return False, f"Need at least {st.session_state.min_train_examples} training examples (have {len(st.session_state.X_train)})."
    X = np.vstack(st.session_state.X_train)
    y = np.array(st.session_state.y_train)
    # sample weighting: recent data more important
    weights = np.linspace(1.0, 3.0, len(y))
    if USE_XGBOOST:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42
        )
        model.fit(X, y, sample_weight=weights)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=42
        )
        model.fit(X, y)
    st.session_state.model = model
    st.session_state.last_trained = time.time()
    return True, "Model trained."

# ---------------------
# Markov probability helper
# ---------------------
def markov_proba(last_seq):
    # try longest match down to length 5
    L = len(last_seq)
    for l in range(L, 4, -1):
        key = tuple(last_seq[-l:])
        counts = st.session_state.markov.get(key, {})
        if counts:
            total = sum(counts.values())
            return [counts.get("D", 0)/total, counts.get("T", 0)/total, counts.get("TIE", 0)/total]
    return None

# ---------------------
# Streak bias heuristic
# ---------------------
def streak_bias_rule(last_seq):
    c = Counter(last_seq)
    if c.get("D", 0) >= int(0.7 * len(last_seq)):  # e.g., 70% same -> bias opposite
        return "T", 64
    if c.get("T", 0) >= int(0.7 * len(last_seq)):
        return "D", 64
    return None, 0

# ---------------------
# Predict now (uses model + markov + streak)
# ---------------------
def predict_now():
    inputs = st.session_state.inputs
    N = st.session_state.context_len
    if len(inputs) < N:
        return None, 0, "need_more"
    lastN = inputs[-N:]
    # streak bias first if model low confidence
    bias_pred, bias_conf = streak_bias_rule(lastN)
    feat = build_features(lastN).reshape(1, -1)
    if st.session_state.model is not None:
        # model proba
        proba = st.session_state.model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = decode_label(pred_idx)
        conf = float(np.max(proba)) * 100
        # combine with markov if present
        m = markov_proba(lastN)
        if m is not None:
            combined = 0.7 * proba + 0.3 * np.array(m)
            pred_idx = int(np.argmax(combined))
            pred_label = decode_label(pred_idx)
            conf = float(np.max(combined)) * 100
        # if model confidence low use bias if available
        if conf < 65 and bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        return pred_label, round(conf), "model"
    else:
        # no model - try markov
        m = markov_proba(lastN)
        if m is not None:
            idx = int(np.argmax(m))
            return decode_label(idx), int(max(m)*100), "markov"
        # fallback frequency
        counts = Counter(lastN)
        best = counts.most_common(1)[0][0]
        return best, 55, "frequency"

# ---------------------
# Layout: Controls
# ---------------------
with st.sidebar:
    st.header("Settings")
    st.subheader("Context (memory) window")
    ctx = st.slider("Context length (how many recent rounds to use)", 10, 20, st.session_state.context_len)
    st.session_state.context_len = ctx

    st.subheader("Training")
    st.session_state.min_train_examples = st.number_input("Min training examples to enable model", min_value=10, max_value=200, value=st.session_state.min_train_examples, step=5)
    st.session_state.auto_retrain = st.checkbox("Auto retrain after new examples", value=st.session_state.auto_retrain)
    st.session_state.auto_mode = st.checkbox("Auto mode (predict next immediately after learning)", value=st.session_state.auto_mode)

    st.markdown("---")
    st.subheader("Model")
    if USE_XGBOOST:
        st.text("Using XGBoost (preferred)")
    else:
        st.text("Using RandomForest (fallback)")

    if os.path.exists("models/dragon_tiger_model.joblib"):
        if st.button("Load saved model"):
            st.session_state.model = joblib.load("models/dragon_tiger_model.joblib")
            st.success("Loaded model from disk")

    if st.button("Rebuild training from inputs"):
        rebuild_training_from_inputs()
        st.success("Rebuilt training set from inputs")

    if st.button("Train model now"):
        ok, msg = train_model(force=True)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

    if st.button("Save model to disk"):
        if st.session_state.model is None:
            st.warning("No model to save.")
        else:
            os.makedirs("models", exist_ok=True)
            joblib.dump(st.session_state.model, "models/dragon_tiger_model.joblib")
            st.success("Model saved to models/dragon_tiger_model.joblib")

# ---------------------
# Main UI: Add input
# ---------------------
st.subheader("🎮 Add Result (D / T / TIE)")
col1, col2 = st.columns([3,1])
with col1:
    choice = st.selectbox("Choose Result", ["D","T","TIE"], key="choice_select")
with col2:
    if st.button("Add Result"):
        st.session_state.inputs.append(choice)
        st.success(f"Added: {choice}")
        # add last training example if enough history
        N = st.session_state.context_len
        i = len(st.session_state.inputs) - 1
        if i >= N:
            hist = st.session_state.inputs[i-N:i]
            st.session_state.X_train.append(build_features(hist))
            st.session_state.y_train.append(encode_label(st.session_state.inputs[i]))
            # update markov counts for keys lengths N down to 5
            for l in range(N, 4, -1):
                if i >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][st.session_state.inputs[i]] += 1
        # auto retrain if enabled and enough data
        if st.session_state.auto_retrain and len(st.session_state.X_train) >= st.session_state.min_train_examples:
            train_model()
        # if auto_mode is off, just add result; if auto_mode on, predict next immediately after learning
        if st.session_state.auto_mode:
            # After adding a real result, predict immediately (the app will show prediction below)
            safe_rerun()

# Developer tools: generate random data / clear
with st.expander("Developer tools"):
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Add 10 random rounds"):
            for _ in range(10):
                st.session_state.inputs.append(random.choice(["D","T","TIE"]))
            rebuild_training_from_inputs()
            st.success("Added 10 random rounds.")
    with c2:
        if st.button("Clear all data"):
            st.session_state.inputs = []
            st.session_state.X_train = []
            st.session_state.y_train = []
            st.session_state.log = []
            st.session_state.markov = defaultdict(lambda: defaultdict(int))
            st.session_state.model = None
            st.session_state.last_trained = None
            st.session_state.loss_streak = 0
            st.success("Cleared everything.")
    with c3:
        if st.button("Force retrain (if enough data)"):
            ok, msg = train_model(force=True)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

# ---------------------
# Prediction panel
# ---------------------
st.markdown("---")
st.subheader("🔮 Predict Next Round")
if len(st.session_state.inputs) < st.session_state.context_len:
    st.info(f"Provide {st.session_state.context_len - len(st.session_state.inputs)} more rounds to begin predictions.")
else:
    pred, conf, source = predict_now()
    st.text(f"Source: {source}")
    st.text(f"Training examples: {len(st.session_state.X_train)}")
    if pred is None:
        st.warning("Prediction unavailable.")
    else:
        if source == "model" and conf >= 65:
            st.success(f"Prediction: **{pred}** | Confidence: {conf}% (model)")
            st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
        elif source == "streak-bias":
            st.warning(f"Prediction (streak-bias): **{pred}** | Confidence: {conf}%")
            st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
        else:
            st.info(f"Prediction (fallback): **{pred}** | Confidence: {conf}%")

        # show training balance
        labels_preview = [decode_label(x) for x in st.session_state.y_train]
        st.text(f"Training Balance ➡️ D: {labels_preview.count('D')} | T: {labels_preview.count('T')} | TIE: {labels_preview.count('TIE')}")

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions in a row. Be cautious!")

        # Confirm & learn block: user provides the actual outcome and app will immediately learn and (if auto_mode) predict the next
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
            # add training example built from the context prior to appending actual
            hist = st.session_state.inputs[-st.session_state.context_len:]
            st.session_state.X_train.append(build_features(hist))
            st.session_state.y_train.append(encode_label(actual))
            # update markov using the current history (before adding actual)
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
            # optionally retrain when enough data
            if st.session_state.auto_retrain and len(st.session_state.X_train) >= st.session_state.min_train_examples:
                train_model()
            # If auto_mode is enabled we want to predict the next immediately: rerun to refresh UI & show updated prediction
            if st.session_state.auto_mode:
                safe_rerun()

# ---------------------
# History / Export
# ---------------------
st.markdown("---")
st.subheader("📈 Prediction History & Exports")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export History to Excel"):
            buf = BytesIO()
            df.to_excel(buf, index=False)
            st.download_button("⬇️ Download Excel", data=buf.getvalue(), file_name="prediction_history.xlsx")
    with col2:
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
    st.info("No history yet. Use 'Confirm & Learn' to add records to history.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ — continuous learning enabled. Tune context window, thresholds, and retrain settings in the sidebar.")
