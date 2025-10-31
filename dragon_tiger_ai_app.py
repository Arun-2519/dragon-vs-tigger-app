import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO

try:
    from sklearn.naive_bayes import MultinomialNB
except:
    st.error("❌ scikit-learn is missing. Add to requirements.txt")
    st.stop()

# -------------------------------- UI & STYLE -------------------------------
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger AI", layout="centered")

st.markdown("""
<style>
body {background: #0A0F24; color: white;}
.stButton>button {
    background: linear-gradient(90deg,#9b27b0,#2196f3);
    color:white;font-weight:bold;border-radius:12px;padding:10px 25px;
}
div[data-testid="stSelectbox"] label, h1, h2, h3, h4 {color:#20e3b2;}
.dataframe {background:white !important;color:black;}
</style>
""", unsafe_allow_html=True)

st.title("🐉 Dragon vs 🌟 Tiger — AI Predictor")

# -------------------------------- STATE INIT -------------------------------
if "inputs" not in st.session_state: st.session_state.inputs = []
if "X_train" not in st.session_state: st.session_state.X_train = []
if "y_train" not in st.session_state: st.session_state.y_train = []
if "log" not in st.session_state: st.session_state.log = []
if "loss_streak" not in st.session_state: st.session_state.loss_streak = 0
if "csv_trained" not in st.session_state: st.session_state.csv_trained = False

# -------------------------------- HELPERS ---------------------------------
def encode(seq):
    m = {"D":0, "T":1, "TIE":2}
    return [m[s] for s in seq if s in m]

def decode(v):
    return {0:"D",1:"T",2:"TIE"}.get(v)

def clean_results(series):
    return series.astype(str).str.upper().str.strip().replace({
        "DRAGON":"D","TIGER":"T","PLAYER":"D","BANKER":"T",
        "DT":"TIE","DRAW":"TIE","TIG":"T","DRA":"D"
    })

def train_model(results):
    results = [r for r in results if r in ["D","T","TIE"]]
    if len(results) < 20: return
    for i in range(10, len(results)):
        seq = results[i-10:i]
        st.session_state.X_train.append(encode(seq))
        st.session_state.y_train.append(encode([results[i]])[0])

def predict(seq):
    if len(seq) < 10: return None, 0
    if len(st.session_state.X_train) < 20: return None, 0
    clf = MultinomialNB()
    w = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=w)
    p = clf.predict([encode(seq[-10:])])[0]
    c = max(clf.predict_proba([encode(seq[-10:])])[0])*100
    return decode(p), round(c,2)

def learn_after_round(seq, actual):
    if len(seq)>=10:
        st.session_state.X_train.append(encode(seq[-10:]))
        st.session_state.y_train.append(encode([actual])[0])

# -------------------------------- CSV Upload -------------------------------
st.subheader("📂 Upload History CSV")
csv = st.file_uploader("Upload D vs T History (Period, Result)", type=["csv"])

if csv and not st.session_state.csv_trained:
    data = pd.read_csv(csv)
    
    if "Result" not in data.columns:
        st.error("❌ CSV must have a 'Result' column")
        st.stop()

    results = clean_results(data["Result"]).tolist()
    
    train_model(results)
    st.session_state.inputs = results[-20:]      # load last rounds to UI
    st.session_state.csv_trained = True
    st.success(f"✅ CSV Loaded & Trained on {len(results)} rounds")

# -------------------------------- USER INPUT -------------------------------
st.subheader("🎮 Add Live Result")

choice = st.selectbox("Choose Result", ["D","T","TIE"])

if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"✅ Added {choice}")

# -------------------------------- PREDICT ---------------------------------
if len(st.session_state.inputs) >= 10:
    pred, conf = predict(st.session_state.inputs)

    if not pred or conf < 55:
        st.warning("⚠️ Collecting patterns... Low confidence")
    else:
        st.success(f"🧠 Prediction: **{pred}** | Confidence: {conf}%")

    # Confirm actual
    actual = st.selectbox("Enter actual result", ["D","T","TIE"])

    if st.button("✅ Confirm & Train"):
        ok = (pred == actual)
        st.session_state.log.append({
            "Prediction":pred, "Confidence":conf,
            "Actual":actual, "Correct":"✅" if ok else "❌"
        })
        learn_after_round(st.session_state.inputs, actual)
        st.session_state.inputs.append(actual)
        st.rerun()

# -------------------------------- HISTORY ---------------------------------
if st.session_state.log:
    st.subheader("📊 Prediction Log")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    if st.button("⬇ Download history"):
        bio = BytesIO()
        df.to_excel(bio, index=False)
        st.download_button("Download Excel", bio.getvalue(),"history.xlsx")
