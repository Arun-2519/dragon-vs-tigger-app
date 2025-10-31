import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO

from sklearn.naive_bayes import MultinomialNB

# ---------------------- UI CONFIG ----------------------
st.set_page_config(page_title="🐉 Dragon vs 🌟 Tiger AI", layout="centered")

st.markdown("""
<style>
body { background: #0b0f19; color: white; font-family: 'Poppins', sans-serif; }
.stButton > button { background: linear-gradient(45deg,#7b2ff7,#f107a3); color:white; border-radius:10px; padding:10px 20px; font-weight:bold; }
div[data-testid="stSelectbox"] label { font-size:18px; color:#f8f8f8; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🐉 Dragon vs 🌟 Tiger — AI Predictor")
st.caption("Hybrid Pattern + Machine Learning Model")

# ---------------------- SESSION ----------------------
for key in ["inputs","X_train","y_train","log","loss_streak","csv_trained"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "loss_streak" else 0

# ---------------------- HELPERS ----------------------
encode = lambda seq: [{'D':0,'T':1,'TIE':2}[x] for x in seq]
decode = lambda v: {0:'D',1:'T',2:'TIE'}[v]

def train_model():
    if len(st.session_state.X_train) < 20: return None
    clf = MultinomialNB()
    w = np.exp(np.linspace(0,1,len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=w)
    return clf

def predict(seq):
    if len(seq) < 10 or len(st.session_state.X_train) < 20:
        return None,0
    clf = train_model(); encoded = encode(seq[-10:])
    pred = clf.predict([encoded])[0]
    prob = clf.predict_proba([encoded])[0]
    return decode(pred), round(max(prob)*100)

def learn(hist, actual):
    if len(hist) >= 10:
        st.session_state.X_train.append(encode(hist[-10:]))
        st.session_state.y_train.append(encode([actual])[0])

# ---------------------- CSV TRAINING ----------------------
st.subheader("📂 Upload Past Game CSV")

csv = st.file_uploader("Upload CSV containing results (D,T,TIE)", type=["csv"])

if csv and not st.session_state.csv_trained:
    data = pd.read_csv(csv)
    results = data.iloc[:,0].astype(str).str.upper().tolist()

    for i in range(10, len(results)):
        hist = results[i-10:i]
        st.session_state.X_train.append(encode(hist))
        st.session_state.y_train.append(encode([results[i]])[0])

    st.session_state.csv_trained = True
    st.success(f"✅ CSV trained! Total patterns: {len(st.session_state.X_train)}")
    st.rerun()

# ---------------------- LIVE INPUT ----------------------
st.subheader("🎮 Add Live Result")
choice = st.selectbox("Select result:", ["D","T","TIE"])

if st.button("Add Result ✅"):
    st.session_state.inputs.append(choice)
    st.success(f"Added {choice}")

    if len(st.session_state.inputs)>=11:
        learn(st.session_state.inputs, choice)

    st.rerun()

# ---------------------- PREDICTION BOX ----------------------
if len(st.session_state.inputs) >= 10 and len(st.session_state.X_train)>=20:
    pred, conf = predict(st.session_state.inputs)

    st.subheader("🤖 AI Prediction")
    st.metric("Prediction", pred, f"{conf}% confidence")

    actual = st.selectbox("Actual result?", ["D","T","TIE"])
    if st.button("Confirm & Train More 🧠"):
        correct = actual == pred
        st.session_state.log.append({"Prediction":pred,"Confidence":conf,"Actual":actual,"Correct":correct})

        learn(st.session_state.inputs, actual)
        st.session_state.inputs.append(actual)

        if correct: st.session_state.loss_streak = 0
        else: st.session_state.loss_streak += 1

        st.rerun()

else:
    st.info("Add minimum 10 live results after training.")

# ---------------------- HISTORY ----------------------
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    buf = BytesIO(); df.to_excel(buf, index=False)
    st.download_button("Download Report 📥", buf.getvalue(), "history.xlsx")
