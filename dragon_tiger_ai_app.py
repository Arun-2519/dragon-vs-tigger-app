# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import base64
from collections import defaultdict
from io import BytesIO
from datetime import datetime

# ---- dependency check ----
try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ Missing dependency: scikit-learn. Add it to requirements.txt and reinstall.")
    st.stop()

# ---- page ----
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger — Predictor", layout="wide", initial_sidebar_state="expanded")
# Custom CSS for nicer look
st.markdown(
    """
    <style>
    :root{
      --bg:#0b0d10;
      --card:#0f1117;
      --muted:#9aa0a6;
      --accent:#9c27b0;
      --accent-2:#7b1fa2;
      --success:#2ecc71;
      --danger:#ff6b6b;
    }
    .reportview-container .main .block-container {
      padding-top: 1rem;
      padding-left: 1.25rem;
      padding-right: 1.25rem;
    }
    .stApp {
      background: linear-gradient(180deg, var(--bg) 0%, #061018 100%);
      color: #e6eef6;
      font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }
    .header {
      display:flex; flex-direction:row; align-items:center; gap:12px;
    }
    .logo-circle {
      width:56px; height:56px; border-radius:12px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      display:flex; align-items:center; justify-content:center; box-shadow: 0 6px 18px rgba(124,58,237,0.12);
      border: 1px solid rgba(255,255,255,0.03);
    }
    .logo-emoji { font-size:24px; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6)); }
    .app-title { font-size:20px; font-weight:700; margin:0; }
    .app-sub { color: var(--muted); margin:0; font-size:13px; }
    .card { background: var(--card); border-radius:12px; padding:14px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.02); }
    .small { color: var(--muted); font-size:13px; }
    .success-pill { background: rgba(46,204,113,0.08); color: var(--success); padding:6px 10px; border-radius:999px; font-weight:600; }
    .danger-pill { background: rgba(255,107,107,0.06); color: var(--danger); padding:6px 10px; border-radius:999px; font-weight:600; }
    /* style download button */
    button[title="Download"] { background: linear-gradient(90deg,var(--accent),var(--accent-2)); border:none; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- header ----
col1, col2 = st.columns([0.18, 0.82])
with col1:
    st.markdown('<div class="logo-circle"><div class="logo-emoji">🐉</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><div><h3 class="app-title">Dragon vs Tiger — Predictor</h3><div class="app-sub">Pattern learning + Naive Bayes + Markov memory — live & CSV import</div></div></div>', unsafe_allow_html=True)
st.markdown("")  # spacer

# ---- session defaults ----
if "inputs" not in st.session_state:
    st.session_state.inputs = []               # user live history (strings 'D','T','TIE')
if "X_train" not in st.session_state:
    st.session_state.X_train = []              # list of encoded 10-length lists
if "y_train" not in st.session_state:
    st.session_state.y_train = []              # labels
if "log" not in st.session_state:
    st.session_state.log = []                  # prediction history
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "markov" not in st.session_state:
    st.session_state.markov = defaultdict(lambda: defaultdict(int))

# ---- helpers ----
LABEL_MAP = {"D": 0, "T": 1, "TIE": 2}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

def encode(seq):
    return [LABEL_MAP[s] for s in seq if s in LABEL_MAP]

def decode(i):
    return INV_MAP.get(i, "")

def safe_append_training(history_list):
    """Append all possible training windows from inputs into X_train/y_train (idempotent)."""
    for i in range(10, len(history_list)):
        window = history_list[i-10:i]
        label = history_list[i]
        enc = encode(window)
        if len(enc) == 10:
            st.session_state.X_train.append(enc)
            st.session_state.y_train.append(encode([label])[0])

# ---- sidebar: upload / settings ----
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Import / Settings")
    uploaded = st.file_uploader("Upload CSV (single column with D/T/TIE)", type=["csv"])
    st.markdown("---")
    st.write("Context window:")
    ctx = st.slider("Rounds to use (timesteps)", 10, 20, 10)
    st.session_state.context_len = ctx
    st.markdown("---")
    st.write("Model controls")
    retrain_btn = st.button("🔁 Rebuild training from history")
    clear_btn = st.button("🧹 Clear all history & models")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- handle uploaded CSV ----
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        # auto-detect likely column
        possible = [c for c in df.columns if c.lower() in ("result","winner","win","outcome")]
        if not possible:
            col = df.columns[0]
        else:
            col = possible[0]
        vals = df[col].astype(str).str.strip().str.upper().tolist()
        # filter / sanitize and keep only D/T/TIE
        vals = [v if v in LABEL_MAP else ("TIE" if v in ("DRAW","PUSH","TIE") else v) for v in vals]
        # keep only D/T/TIE values (fallback: try to map first char)
        cleaned = []
        for v in vals:
            if v in LABEL_MAP:
                cleaned.append(v)
            else:
                if len(v)>0 and v[0] in ("D","T"):
                    cleaned.append("D" if v[0]=="D" else "T")
        if not cleaned:
            st.error("No valid D/T/TIE values detected in CSV.")
        else:
            st.success(f"Loaded {len(cleaned)} rounds from CSV (column: {col})")
            # seed session history with CSV history but do NOT overwrite existing live inputs
            if len(st.session_state.inputs) == 0:
                st.session_state.inputs = cleaned.copy()
                # build training arrays
                safe_append_training(st.session_state.inputs)
            else:
                # If user already had live history - offer choice to merge
                if st.button("Merge CSV into existing history"):
                    st.session_state.inputs = st.session_state.inputs + cleaned
                    safe_append_training(st.session_state.inputs)
                    st.success("Merged CSV into live history.")
    except Exception as e:
        st.error(f"CSV load error: {e}")

# ---- handle sidebar actions ----
if retrain_btn:
    # rebuild X_train/y_train from current session inputs
    st.session_state.X_train = []
    st.session_state.y_train = []
    safe_append_training(st.session_state.inputs)
    st.sidebar.success("Rebuilt training from history")

if clear_btn:
    st.session_state.inputs = []
    st.session_state.X_train = []
    st.session_state.y_train = []
    st.session_state.log = []
    st.session_state.loss_streak = 0
    st.session_state.markov = defaultdict(lambda: defaultdict(int))
    st.sidebar.success("Cleared all session data")

# ---- main layout columns ----
colA, colB = st.columns([0.56, 0.44])

# left column: input / predict
with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎮 Live Input")
    st.markdown('<div class="small">Add the most recent round result — the system will continuously learn.</div>', unsafe_allow_html=True)
    choice = st.selectbox("Add result", ["D", "T", "TIE"], index=0)
    if st.button("➕ Add Result"):
        st.session_state.inputs.append(choice)
        # update markov memory & immediate training append if possible
        i = len(st.session_state.inputs) - 1
        N = st.session_state.context_len if "context_len" in st.session_state else 10
        if i >= N:
            hist_slice = st.session_state.inputs[i-N:i]
            st.session_state.X_train.append(encode(hist_slice))
            st.session_state.y_train.append(encode([st.session_state.inputs[i]])[0])
            # update markov counts
            for l in range(N, 4, -1):
                if i >= l:
                    key = tuple(st.session_state.inputs[i-l:i])
                    st.session_state.markov[key][st.session_state.inputs[i]] += 1
        st.success(f"Added: {choice}")
        st.experimental_rerun()

    st.markdown("---")

    # prediction area
    st.subheader("🔮 Prediction")
    def can_predict():
        return len(st.session_state.inputs) >= 10 and len(st.session_state.X_train) >= 20

    def predict_nb(seq):
        # original NB predictor with exponential weighting
        encoded = encode(seq[-10:])
        clf = MultinomialNB()
        weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train))) if len(st.session_state.X_train)>0 else None
        try:
            clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
            p = clf.predict([encoded])[0]
            prob = max(clf.predict_proba([encoded])[0]) * 100
            return decode(p), round(prob)
        except Exception as e:
            return None, 0

    if not can_predict():
        need_rounds = max(0, 10 - len(st.session_state.inputs))
        need_patterns = max(0, 20 - len(st.session_state.X_train))
        st.info(f"Need {need_rounds} more recent rounds and {need_patterns} learned patterns to start predicting.")
    else:
        pred, conf = predict_nb(st.session_state.inputs)
        if pred is None or conf < 65:
            st.warning("⚠️ Low confidence or insufficient patterns. The predictor will keep learning.")
        else:
            st.metric(label="AI Prediction", value=f"{pred}", delta=f"{conf}% confidence")
            if st.session_state.loss_streak >= 3:
                st.warning("⚠️ Recent loss streak detected — predictions may be unreliable.")

        # Confirm actual
        st.markdown("**Confirm actual outcome (teach & continue)**")
        actual = st.selectbox("Actual result (confirm)", ["D", "T", "TIE"], index=0, key="confirm_actual")
        if st.button("✅ Confirm & Learn"):
            correct = (actual == pred)
            st.session_state.log.append({
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": pred or "N/A",
                "confidence": conf,
                "actual": actual,
                "correct": "✅" if correct else "❌"
            })
            # append training sample & update markov
            i = len(st.session_state.inputs)
            N = st.session_state.context_len
            if i >= N:
                hist_slice = st.session_state.inputs[i-N:i]
                st.session_state.X_train.append(encode(hist_slice))
                st.session_state.y_train.append(encode([actual])[0])
                for l in range(N, 4, -1):
                    if i >= l:
                        key = tuple(st.session_state.inputs[i-l:i])
                        st.session_state.markov[key][actual] += 1
            st.session_state.inputs.append(actual)
            if correct:
                st.session_state.loss_streak = 0
            else:
                st.session_state.loss_streak += 1
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# right column: stats / history / downloads
with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Training & History")
    st.markdown("<div class='small'>Quick session stats and export tools</div>", unsafe_allow_html=True)

    # training balance
    labels = [decode(y) for y in st.session_state.y_train]
    d_count = labels.count("D")
    t_count = labels.count("T")
    tie_count = labels.count("TIE")
    st.write(f"Training patterns: **{len(st.session_state.X_train)}**")
    cols = st.columns(3)
    cols[0].metric("Dragon (D)", d_count)
    cols[1].metric("Tiger (T)", t_count)
    cols[2].metric("Tie", tie_count)

    st.markdown("---")
    st.write("Recent rounds (most recent last):")
    recent = st.session_state.inputs[-30:]
    if recent:
        st.code("  ".join(recent[-30:]))
    else:
        st.info("No rounds recorded yet.")

    if st.session_state.log:
        st.markdown("---")
        st.write("Prediction log (last 20):")
        df_log = pd.DataFrame(st.session_state.log[-20:]).iloc[::-1]  # newest first
        st.dataframe(df_log)

        buf = BytesIO()
        df_log.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button("⬇️ Download Log (Excel)", data=buf.getvalue(), file_name="prediction_history.xlsx")

    st.markdown("---")
    st.write("Utilities")
    if st.button("🗑️ Reset Loss Streak"):
        st.session_state.loss_streak = 0
        st.success("Loss streak reset.")

    # small help
    st.markdown("<div class='small' style='margin-top:8px'>Tip: upload a CSV of past rounds (Result column). Then press 'Rebuild training from history' in sidebar to retrain patterns.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# footer / credits
st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div class="small">Built with ❤️ — Naive Bayes + Markov memory</div><div class="small">Version: 1.1</div></div>', unsafe_allow_html=True)
