import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB

# ✅ UI Styling
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger AI", layout="centered")
st.markdown("""
<style>
body { background-color: #0f1117; color: white; }
.stButton>button { background-color: #ff5c93; color: white; font-weight: bold; border-radius: 8px; }
.sidebar .sidebar-content { background: #111418; }
.css-1v3fvcr { background:#0f1117 !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🐉 Dragon vs 🌟 Tiger — AI Predictor")

# Session States
if "inputs" not in st.session_state: st.session_state.inputs = []
if "X_train" not in st.session_state: st.session_state.X_train = []
if "y_train" not in st.session_state: st.session_state.y_train = []
if "data_loaded" not in st.session_state: st.session_state.data_loaded = False
if "log" not in st.session_state: st.session_state.log = []
if "loss_streak" not in st.session_state: st.session_state.loss_streak = 0
if "model_ready" not in st.session_state: st.session_state.model_ready = False

# Encoding
encode_map = {'D': 0, 'T': 1, 'TIE': 2}
decode_map = {0: 'D', 1: 'T', 2: 'TIE'}

def encode(seq):
    return [encode_map[s] for s in seq]

def decode(v):
    return decode_map.get(v, "")

# Train Model
def train_model():
    if len(st.session_state.X_train) < 20:
        return None
    
    clf = MultinomialNB()
    weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
    return clf

# Predict
def predict(seq):
    if len(seq) < 10 or len(st.session_state.X_train) < 20:
        return None, 0
    
    clf = train_model()
    if clf is None: return None, 0
    
    encoded = encode(seq[-10:])
    pred = clf.predict([encoded])[0]
    conf = max(clf.predict_proba([encoded])[0]) * 100
    return decode(pred), round(conf)

# Add training pattern
def push_training(history, result):
    if len(history) < 10: return
    st.session_state.X_train.append(encode(history[-10:]))
    st.session_state.y_train.append(encode([result])[0])

# ✅ CSV Upload & Train
st.subheader("📂 Upload Past Game CSV (Period, Result)")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    if df.shape[1] >= 2:
        results = df.iloc[:, 1].astype(str).str.upper()
        results = results.replace({"DRAGON": "D", "TIGER": "T"})
        results = results[results.isin(["D", "T", "TIE"])]

        st.session_state.inputs = results.tolist()

        # Train from CSV
        st.session_state.X_train, st.session_state.y_train = [], []
        for i in range(10, len(st.session_state.inputs)):
            push_training(st.session_state.inputs[:i], st.session_state.inputs[i])

        st.success(f"✅ Training finished with {len(st.session_state.X_train)} patterns")
        st.session_state.model_ready = True
    else:
        st.error("CSV must have at least 2 columns: Period, Result")

# ✅ Live Input
st.subheader("🎮 Add Live Result")
choice = st.selectbox("Select Result", ["D", "T", "TIE"])
if st.button("Add Result"):
    st.session_state.inputs.append(choice)
    push_training(st.session_state.inputs[:-1], choice)
    st.success(f"Added: {choice}")

# ✅ Prediction
if len(st.session_state.inputs) >= 10 and st.session_state.model_ready:
    pred, conf = predict(st.session_state.inputs)

    if pred:
        if conf >= 65:
            st.success(f"✅ Prediction: **{pred}** | Confidence: {conf}%")
        else:
            st.warning(f"⚠️ Low Confidence ({conf}%) — Continue input")
    else:
        st.info("💡 Keep adding data to start prediction.")

# ✅ History
if st.session_state.log:
    st.subheader("📊 History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    if st.button("Download Log Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇ Download", data=buf.getvalue(), file_name="history.xlsx")

st.caption("⚡ Powered by Pattern AI + Naive Bayes")
