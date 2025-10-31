import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from io import BytesIO

# ✅ Check sklearn
try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ Install scikit-learn in requirements.txt")
    st.stop()

# ================= UI SETTINGS =================
st.set_page_config(page_title="🐉⚖️🌟 Dragon Tiger AI", layout="centered")

st.title("🐉 Dragon vs 🌟 Tiger Predictor (AI Powered)")
st.markdown("""
<style>
body { background:#0f1117; color:#fff; }
.stButton>button { background:#9c27b0; color:white; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ================= STATE =================
if "inputs" not in st.session_state: st.session_state.inputs=[]
if "X_train" not in st.session_state: st.session_state.X_train=[]
if "y_train" not in st.session_state: st.session_state.y_train=[]
if "log" not in st.session_state: st.session_state.log=[]
if "loss_streak" not in st.session_state: st.session_state.loss_streak=0
if "markov" not in st.session_state: st.session_state.markov=defaultdict(lambda: defaultdict(int))

# ================= HELPERS =================
def encode(seq):
    m={'D':0,'T':1,'TIE':2}
    return [m[s] for s in seq if s in m]

def decode(v): 
    return {0:'D',1:'T',2:'TIE'}.get(v,"")

# ================= PREDICT =================
def predict(seq):
    if len(seq) < 10: return None,0
    if len(st.session_state.X_train) < 20: return None,0

    encoded = encode(seq[-10:])
    model = MultinomialNB()
    weights = np.exp(np.linspace(0,1,len(st.session_state.X_train)))

    model.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)

    pred = model.predict([encoded])[0]
    conf = float(model.predict_proba([encoded])[0].max()) * 100
    return decode(pred), round(conf)

# ================= LEARN =================
def learn(seq, actual):
    if len(seq) >= 10:
        st.session_state.X_train.append(encode(seq[-10:]))
        st.session_state.y_train.append(encode([actual])[0])

# ============ ✅ CSV UPLOAD + TRAINING ==============
st.subheader("📂 Import Past Dragon-Tiger Results (Optional)")
uploaded = st.file_uploader("Upload CSV containing D/T/TIE results", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    
    # Expect column names
    possible_cols = ["Result","result","WIN","winner","outcome"]
    col = next((c for c in possible_cols if c in df.columns), None)

    if not col:
        st.error("CSV must have a column: Result / winner / WIN / outcome")
    else:
        st.success("✅ Training model from CSV history")

        seq = df[col].astype(str).str.upper().tolist()

        for i in range(10, len(seq)):
            X = encode(seq[i-10:i])
            y = encode([seq[i]])[0]
            if len(X)==10:
                st.session_state.X_train.append(X)
                st.session_state.y_train.append(y)

        st.info(f"✅ Loaded {len(st.session_state.X_train)} training patterns from CSV")

# ================ USER INPUT =================
st.subheader("🎮 Add Result (D / T / TIE)")
choice = st.selectbox("Choose Result", ["D","T","TIE"])
if st.button("Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"✅ Added: {choice}")

# Continue learning from live data
if len(st.session_state.inputs) > 10:
    for i in range(10,len(st.session_state.inputs)):
        X = encode(st.session_state.inputs[i-10:i])
        y = encode([st.session_state.inputs[i]])[0]
        if len(X)==10:
            st.session_state.X_train.append(X)
            st.session_state.y_train.append(y)

# ================= RUN PREDICTION =================
if len(st.session_state.inputs) >= 10:
    pred, conf = predict(st.session_state.inputs)

    labels = [decode(y) for y in st.session_state.y_train]
    st.text(f"Training ➡️ D: {labels.count('D')} | T: {labels.count('T')} | TIE: {labels.count('TIE')}")

    if pred is None or conf < 65:
        st.warning("⚠️ Low confidence / not enough data")
    else:
        st.success(f"🧠 Prediction: **{pred}** | Confidence: {conf}%")

    actual = st.selectbox("Enter actual result:", ["D","T","TIE"])

    if st.button("Confirm & Learn"):
        correct = actual == pred
        st.session_state.log.append({
            "Prediction": pred,
            "Confidence": conf,
            "Actual": actual,
            "Correct": "✅" if correct else "❌"
        })

        learn(st.session_state.inputs, actual)
        st.session_state.inputs.append(actual)
        st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak+1
        st.rerun()

else:
    st.info(f"Need {10-len(st.session_state.inputs)} more inputs to start predicting.")

# ================= HISTORY =================
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)

    if st.button("Download History"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", buf.getvalue(), "prediction_history.xlsx")

st.caption("Built with ❤️ | Naive Bayes + Pattern Memory")
