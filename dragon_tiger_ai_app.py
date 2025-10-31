# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter, deque
from io import BytesIO
import joblib
import os
import time

# Try xgboost, fallback to RandomForestClassifier
USE_XGBOOST = False
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier

# basic page setup
st.set_page_config(page_title="🐉⚖️🌟 Dragon Tiger AI", layout="centered")
st.title("🐉 Dragon vs 🌟 Tiger Predictor (AI Powered)")

st.markdown(
    """
    <style>
        body { background-color: #0f1117; color: #ffffff; }
        .stButton>button {
            background-color: #9c27b0;
            color: white;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #1976d2;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State initialization ---
if "inputs" not in st.session_state:
    st.session_state.inputs = []                # full sequence of rounds (strings 'D','T','TIE')
if "X_train" not in st.session_state:
    st.session_state.X_train = []               # list of feature vectors
if "y_train" not in st.session_state:
    st.session_state.y_train = []               # list of labels encoded 0/1/2
if "log" not in st.session_state:
    st.session_state.log = []                   # prediction history logs
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))
if "model" not in st.session_state:
    st.session_state.model = None
if "last_trained" not in st.session_state:
    st.session_state.last_trained = None

# Encoding helpers
LABEL_MAP = {'D': 0, 'T': 1, 'TIE': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def encode_label(s):
    return LABEL_MAP.get(s, -1)

def decode_label(i):
    return INV_LABEL_MAP.get(i, "")

# Feature builder
def build_features(history_last_n):
    """
    history_last_n: list of last 10 outcomes strings, e.g. ['D','T',...]
    returns: numeric feature vector (1D)
    Features included:
      - encoded last 10 (as ints)
      - count_D, count_T, count_TIE (3)
      - last_winner (0/1/2)
      - last_streak_length (int)
      - proportion_D_in_last10, proportion_T_in_last10
    Total length = 10 + 3 + 1 + 1 + 2 = 17
    """
    last10 = history_last_n[-10:]
    encoded = [LABEL_MAP.get(x, 2) for x in last10]  # default map
    c = Counter(last10)
    count_D = c.get('D', 0)
    count_T = c.get('T', 0)
    count_TIE = c.get('TIE', 0)
    last_winner = LABEL_MAP.get(last10[-1], 2)
    # compute last streak length
    streak_len = 1
    for i in range(len(last10)-2, -1, -1):
        if last10[i] == last10[-1]:
            streak_len += 1
        else:
            break
    prop_D = count_D / len(last10)
    prop_T = count_T / len(last10)
    features = encoded + [count_D, count_T, count_TIE, last_winner, streak_len, prop_D, prop_T]
    return np.array(features, dtype=float)

# Build training rows from current inputs (used when user adds)
def add_training_from_inputs():
    inputs = st.session_state.inputs
    # For i starting from 10 -> len(inputs)-1, create features from i-10..i-1 and label inputs[i]
    for i in range(10, len(inputs)):
        history_slice = inputs[i-10:i]
        result = inputs[i]
        # create feature vector
        f = build_features(history_slice)
        # avoid duplicates: a simple rule is to append all; duplicates OK for model
        st.session_state.X_train.append(f)
        st.session_state.y_train.append(encode_label(result))
        # update markov counts
        for l in range(10, 4, -1):
            if len(inputs[:i]) >= l:
                key = tuple(inputs[i-l:i])
                st.session_state.markov[key][result] += 1

# Train / (re)fit model function
def train_model(force=False):
    if len(st.session_state.X_train) < 40 and not force:
        # require more patterns for stable training
        return False, "Need at least 40 training examples to train the model."
    X = np.vstack(st.session_state.X_train)
    y = np.array(st.session_state.y_train)
    # sample weighting: give more weight to recent examples
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
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=42
        )
    model.fit(X, y, sample_weight=weights) if USE_XGBOOST else model.fit(X, y)
    st.session_state.model = model
    st.session_state.last_trained = time.time()
    return True, "Model trained."

# Predict function uses model + simple streak-bias fallback
def streak_bias_rule(last10):
    # if extreme streaks present, bias to opposite (simple heuristic)
    c = Counter(last10)
    if c.get('D', 0) >= 7:
        return 'T', 64
    if c.get('T', 0) >= 7:
        return 'D', 64
    return None, 0

def markov_proba(last_k):
    # return distribution from stored markov counts for the best matching key
    # try longest matching key first
    for l in range(len(last_k), 4, -1):
        key = tuple(last_k[-l:])
        counts = st.session_state.markov.get(key, {})
        if counts:
            total = sum(counts.values())
            probs = {k: v/total for k, v in counts.items()}
            # return probabilities as list in order [D,T,TIE]
            return [probs.get('D',0), probs.get('T',0), probs.get('TIE',0)]
    return None

def predict_now():
    inputs = st.session_state.inputs
    if len(inputs) < 10:
        return None, 0, "Need 10 rounds"
    # If model exists use it, otherwise fallback
    last10 = inputs[-10:]
    # check simple streak bias first
    bias_pred, bias_conf = streak_bias_rule(last10)
    # build features
    feat = build_features(last10).reshape(1, -1)
    # If we have a trained model:
    if st.session_state.model is not None:
        proba = st.session_state.model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = decode_label(pred_idx)
        conf = float(np.max(proba)) * 100
        # incorporate markov info if available (weighted avg)
        markov_p = markov_proba(last10)
        if markov_p is not None:
            # combine model proba and markov with weights (70% model, 30% markov)
            combined = 0.7 * proba + 0.3 * np.array(markov_p)
            pred_idx = int(np.argmax(combined))
            pred_label = decode_label(pred_idx)
            conf = float(np.max(combined)) * 100
        # if model low confidence, use streak bias if exists
        if conf < 65 and bias_pred:
            return bias_pred, bias_conf, "streak-bias"
        return pred_label, round(conf), "model"
    else:
        # no model: if we have markov, use it; else fallback to frequency
        markov_p = markov_proba(last10)
        if markov_p is not None:
            idx = int(np.argmax(markov_p))
            return decode_label(idx), int(max(markov_p)*100), "markov"
        # fallback frequency
        counts = Counter(last10)
        best = counts.most_common(1)[0][0]
        return best, 55, "frequency"

# --- Input UI ---
st.subheader("🎮 Add Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D", "T", "TIE"])
if st.button("Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added: {choice}")
    # add new train rows if possible
    if len(st.session_state.inputs) > 10:
        # just add the last training point to avoid re-adding everything repeatedly
        i = len(st.session_state.inputs) - 1
        if i >= 10:
            history_slice = st.session_state.inputs[i-10:i]
            result = st.session_state.inputs[i]
            st.session_state.X_train.append(build_features(history_slice))
            st.session_state.y_train.append(encode_label(result))
            # update markov
            for l in range(10, 4, -1):
                if len(st.session_state.inputs[:i]) >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][result] += 1

# --- Bulk add example / randomize (developer helper) ---
with st.expander("Developer tools (Generate sample/random data)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add 10 random rounds"):
            for _ in range(10):
                st.session_state.inputs.append(random.choice(["D","T","TIE"]))
            st.success("Added 10 random rounds.")
    with col2:
        if st.button("Clear inputs"):
            st.session_state.inputs = []
            st.session_state.X_train = []
            st.session_state.y_train = []
            st.session_state.log = []
            st.session_state.markov = defaultdict(lambda: defaultdict(int))
            st.session_state.model = None
            st.success("Cleared all data.")
    with col3:
        if st.button("Force retrain (if enough data)"):
            ok, msg = train_model(force=True)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

# --- Build training set from inputs if not already (first time) ---
# (This will only run when we have unaccounted inputs and X_train is small)
if len(st.session_state.X_train) < max(0, len(st.session_state.inputs) - 10):
    add_training_from_inputs()

# --- Show dataset counts and train button ---
st.markdown("---")
st.subheader("📊 Training Data & Model")
st.text(f"Rounds recorded: {len(st.session_state.inputs)}")
st.text(f"Training examples: {len(st.session_state.X_train)}")
labels_preview = [decode_label(x) for x in st.session_state.y_train]
st.text(f"Training Balance ➡️ D: {labels_preview.count('D')} | T: {labels_preview.count('T')} | TIE: {labels_preview.count('TIE')}")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Train Model"):
        ok, msg = train_model(force=False)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
with col_b:
    if st.button("Save Model to disk"):
        if st.session_state.model is None:
            st.warning("No trained model to save.")
        else:
            os.makedirs("models", exist_ok=True)
            joblib.dump(st.session_state.model, "models/dragon_tiger_model.joblib")
            st.success("Model saved to models/dragon_tiger_model.joblib")

# Allow user to load saved model
if os.path.exists("models/dragon_tiger_model.joblib"):
    if st.button("Load saved model"):
        st.session_state.model = joblib.load("models/dragon_tiger_model.joblib")
        st.success("Loaded model from models/dragon_tiger_model.joblib")

# --- Prediction Section ---
st.markdown("---")
st.subheader("🔮 Prediction")
if len(st.session_state.inputs) < 10:
    st.info(f"Enter {10 - len(st.session_state.inputs)} more inputs to begin prediction.")
else:
    pred, conf, source = predict_now()
    # show debug info
    st.text(f"Prediction source: {source}")
    if pred is None:
        st.warning("Not enough data or model not trained.")
    else:
        if source == "model" and conf >= 65:
            st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
            st.subheader("🧠 AI Prediction")
            st.success(f"Prediction: **{pred}** | Confidence: {conf}%")
        elif source == "streak-bias":
            st.warning("⚠️ Low model confidence — applying streak-bias heuristic.")
            st.warning(f"Prediction (streak-bias): **{pred}** | Confidence: {conf}%")
            st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
        else:
            st.info(f"Prediction (fallback): **{pred}** | Confidence: {conf}%")

        if st.session_state.loss_streak >= 3:
            st.warning("⚠️ Multiple wrong predictions. Be cautious!")

        # confirm & learn
        actual = st.selectbox("Enter actual result", ["D", "T", "TIE"], key="confirm_actual")
        if st.button("Confirm & Learn"):
            correct = actual == pred
            st.session_state.log.append({
                "Prediction": pred,
                "Confidence": conf,
                "Source": source,
                "Actual": actual,
                "Correct": "✅" if correct else "❌",
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            # add new training example using last 10 history (before actual appended)
            if len(st.session_state.inputs) >= 10:
                st.session_state.X_train.append(build_features(st.session_state.inputs[-10:]))
                st.session_state.y_train.append(encode_label(actual))
            # update markov counts
            i = len(st.session_state.inputs)
            for l in range(10, 4, -1):
                if i >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][actual] += 1
            # append actual to history
            st.session_state.inputs.append(actual)
            if correct:
                st.session_state.loss_streak = 0
            else:
                st.session_state.loss_streak += 1
            # optionally retrain automatically when enough new samples exist
            if len(st.session_state.X_train) >= 60:
                train_model()
            st.experimental_rerun()

# --- History & download ---
st.markdown("---")
st.subheader("📈 Prediction History")
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
                cols = [f"enc_{i+1}" for i in range(10)] + ["count_D","count_T","count_TIE","last_winner","streak_len","prop_D","prop_T"]
                tr_df = pd.DataFrame(X, columns=cols)
                tr_df["label"] = [decode_label(int(x)) for x in y]
                buf = BytesIO()
                tr_df.to_csv(buf, index=False)
                st.download_button("⬇️ Download training CSV", data=buf.getvalue(), file_name="training_set.csv")
            else:
                st.warning("No training data to export.")
else:
    st.info("No prediction history yet.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, XGBoost/RandomForest, and pattern learning. Tweak the training thresholds and hyperparameters in-app or in repo.")

