# lottery7_streamlit_nb_simple.py
# Lightweight Lottery7 Wingo app (Markov + Frequency + pattern ensemble)
# Uses the Dragon-Tiger example style for continuous learning. Reference: uploaded example. :contentReference[oaicite:1]{index=1}

import streamlit as st
import numpy as np
import pandas as pd
import json, os, random
from collections import defaultdict, deque
from io import BytesIO

# optional sklearn import for simple NB fallback (not required)
try:
    from sklearn.naive_bayes import MultinomialNB
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="🎯 Lottery7 Wingo — Simple Predictor", layout="centered")
st.title("🎯 Lottery7 Wingo — Simple Predictor (Markov + Frequency)")

# ------------------------
# Config & rules
# ------------------------
WINDOW = st.sidebar.number_input("Window (recent rounds)", value=10, min_value=3, max_value=30)
CONF_THRESHOLD = st.sidebar.slider("Recommendation confidence %", 0, 100, 65)
W_MARKOV = st.sidebar.slider("Weight: Markov", 0.0, 1.0, 0.6)
W_FREQ = st.sidebar.slider("Weight: Freq", 0.0, 1.0, 0.3)
W_PATTERN = st.sidebar.slider("Weight: RecentPattern", 0.0, 1.0, 0.1)

NUM_PLACES = 3
NUM_CLASSES_NUM = 10
NUM_CLASSES_COL = 3  # 0=G,1=R,2=V
INV_COLOR = {0:'G',1:'R',2:'V'}

GREEN_NUMS = {1,3,7,9}
RED_NUMS = {2,4,6,8}
AMBIGUOUS_NUMS = {0,5}

HISTORY_PATH = "history_rules.json"

# ------------------------
# Helpers: Markov & Frequency
# ------------------------
class Markov:
    def __init__(self, n):
        self.n = n
        # initialize uniform+1 smoothing
        self.counts = np.ones((n, n), dtype=np.float32)
    def update(self, prev, nxt):
        self.counts[int(prev), int(nxt)] += 1
    def predict(self, last):
        row = self.counts[int(last)].astype(np.float32)
        return row / row.sum()

class Frequency:
    def __init__(self, n):
        self.counts = np.ones(n, dtype=np.float32)
    def update(self, x):
        self.counts[int(x)] += 1
    def prob(self):
        return self.counts / self.counts.sum()

# ------------------------
# Session state / history
# ------------------------
if 'history' not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        try:
            st.session_state.history = json.load(open(HISTORY_PATH,'r'))
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

# initialize priors
if 'markov_num' not in st.session_state:
    st.session_state.markov_num = [Markov(NUM_CLASSES_NUM) for _ in range(NUM_PLACES)]
if 'freq_num' not in st.session_state:
    st.session_state.freq_num = [Frequency(NUM_CLASSES_NUM) for _ in range(NUM_PLACES)]
if 'freq_col' not in st.session_state:
    st.session_state.freq_col = [Frequency(NUM_CLASSES_COL) for _ in range(NUM_PLACES)]

if 'pending' not in st.session_state:
    st.session_state.pending = None
if 'log' not in st.session_state:
    st.session_state.log = []

# populate priors from existing history once
if not st.session_state.get('_priors_populated', False):
    hist = st.session_state.history
    for i in range(len(hist)-1):
        cur = hist[i]; nxt = hist[i+1]
        for p in range(NUM_PLACES):
            st.session_state.markov_num[p].update(cur['places'][p]['num'], nxt['places'][p]['num'])
            st.session_state.freq_num[p].update(nxt['places'][p]['num'])
            st.session_state.freq_col[p].update(nxt['places'][p]['color'])
    st.session_state._priors_populated = True

# ------------------------
# Encoders + utils
# ------------------------
def save_history():
    json.dump(st.session_state.history, open(HISTORY_PATH,'w'), indent=2)

def round_to_display(r):
    return ','.join([f"{p['num']}{INV_COLOR[p['color']]}{'B' if p.get('size_observed', p['num']>=5) else 'S'}" for p in r['places']])

# ------------------------
# UI: Minimal 3 inputs per place
# ------------------------
st.subheader("Enter new round (Size, Color, Number) — minimal input")
cols = st.columns(NUM_PLACES)
place_inputs = []
COLOR_CHOICES = ['G','R','V','R+V','G+V']
SIZE_CHOICES = ['S','B']

for i in range(NUM_PLACES):
    with cols[i]:
        st.markdown(f"**Place P{i+1}**")
        size_choice = st.selectbox(f"P{i+1} Size", options=SIZE_CHOICES, index=0, key=f'size_in_{i}')
        col_choice = st.selectbox(f"P{i+1} Color", options=COLOR_CHOICES, index=0, key=f'col_in_{i}')
        num_choice = st.number_input(f"P{i+1} Number", min_value=0, max_value=9, value=0, key=f'num_in_{i}')
        place_inputs.append({'num': int(num_choice), 'color_raw': col_choice, 'size_raw': size_choice})

if st.button("Queue round"):
    # map color_raw -> primary int + ambiguous flag
    pending = {'places': []}
    for p in range(NUM_PLACES):
        n = place_inputs[p]['num']
        c_raw = place_inputs[p]['color_raw']
        s_raw = place_inputs[p]['size_raw']
        if c_raw == 'G':
            c_int = 0; amb = None
        elif c_raw == 'R':
            c_int = 1; amb = None
        elif c_raw == 'V':
            c_int = 2; amb = None
        elif c_raw == 'R+V':
            c_int = 1; amb = 'R+V'
        elif c_raw == 'G+V':
            c_int = 0; amb = 'G+V'
        else:
            c_int = 1; amb = None
        s_int = 1 if s_raw == 'B' else 0
        pending['places'].append({
            'num': int(n),
            'color': int(c_int),
            'color_raw': c_raw,
            'ambiguous_color': amb,
            'size_observed': int(s_int),
            'size_override_used': True
        })
    st.session_state.pending = pending
    st.success("Queued round — confirm to learn")

if st.session_state.pending:
    st.info("Pending: " + round_to_display(st.session_state.pending))

# ------------------------
# Prediction core (ensemble: Markov + Freq + Recent pattern)
# ------------------------
def predict_from_history():
    hist = st.session_state.history
    L = len(hist)
    results = {}
    if L < 1:
        # return uniform defaults if no history
        for p in range(NUM_PLACES):
            results[p] = {
                'num_probs': np.ones(NUM_CLASSES_NUM)/NUM_CLASSES_NUM,
                'col_probs': np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL,
                'size_probs': np.array([0.5,0.5])
            }
        return results

    # for each place compute ensemble
    for p in range(NUM_PLACES):
        # 1) Markov: use last observed number for that place if available
        if L >= 1:
            last_num = hist[-1]['places'][p]['num']
            markov_prob = st.session_state.markov_num[p].predict(last_num)
        else:
            markov_prob = np.ones(NUM_CLASSES_NUM)/NUM_CLASSES_NUM

        # 2) Frequency prior
        freq_prob = st.session_state.freq_num[p].prob()

        # 3) recent-window pattern: count occurrences of numbers in last WINDOW
        recent_counts = np.ones(NUM_CLASSES_NUM, dtype=np.float32)  # smoothing
        start = max(0, L - WINDOW)
        for i in range(start, L):
            val = hist[i]['places'][p]['num']
            recent_counts[int(val)] += 1
        recent_prob = recent_counts / recent_counts.sum()

        # combine with weights (normalized)
        w_total = W_MARKOV + W_FREQ + W_PATTERN
        if w_total <= 0:
            w_total = 1.0
        combined_num = (W_MARKOV*markov_prob + W_FREQ*freq_prob + W_PATTERN*recent_prob) / w_total

        # color: derive from numbers mostly. For ambiguous numbers (0,5) consult color frequency
        col_probs = np.zeros(NUM_CLASSES_COL, dtype=np.float32)
        # get color_model split from frequency for ambiguous case
        color_freq = st.session_state.freq_col[p].prob()
        for n in range(NUM_CLASSES_NUM):
            pn = combined_num[n]
            if n in GREEN_NUMS:
                col_probs[0] += pn
            elif n in RED_NUMS:
                col_probs[1] += pn
            elif n in AMBIGUOUS_NUMS:
                # split ambiguous probability using color_freq's red/violet portion
                red_share = color_freq[1]
                vio_share = color_freq[2]
                total = red_share + vio_share
                if total <= 0:
                    col_probs[1] += pn*0.5
                    col_probs[2] += pn*0.5
                else:
                    col_probs[1] += pn * (red_share/total)
                    col_probs[2] += pn * (vio_share/total)
            else:
                col_probs += pn / NUM_CLASSES_COL
        if col_probs.sum() > 0:
            col_probs /= col_probs.sum()
        else:
            col_probs = np.ones(NUM_CLASSES_COL)/NUM_CLASSES_COL

        # size derived deterministically from number probabilities
        small_prob = combined_num[:5].sum()
        big_prob = combined_num[5:].sum()
        size_probs = np.array([small_prob, big_prob], dtype=np.float32)
        if size_probs.sum() > 0:
            size_probs = size_probs / size_probs.sum()
        else:
            size_probs = np.array([0.5,0.5])

        results[p] = {
            'num_probs': combined_num,
            'col_probs': col_probs,
            'size_probs': size_probs
        }
    return results

# ------------------------
# Live prediction UI
# ------------------------
st.subheader("Live prediction (based on current history)")
if len(st.session_state.history) >= max(1, WINDOW//2):
    preds = predict_from_history()
    place_choice = st.selectbox('Select place to inspect / bet', options=['P1','P2','P3'])
    pi = int(place_choice[1]) - 1
    num_probs = preds[pi]['num_probs']
    col_probs = preds[pi]['col_probs']
    size_probs = preds[pi]['size_probs']

    pred_num = int(np.argmax(num_probs)); conf_num = float(np.max(num_probs))*100.0
    pred_col = int(np.argmax(col_probs)); conf_col = float(np.max(col_probs))*100.0
    pred_size_idx = int(np.argmax(size_probs)); conf_size = float(np.max(size_probs))*100.0

    st.write(f"Place {place_choice} prediction:")
    st.write(f"- Number: {pred_num} (conf {conf_num:.1f}%)")
    st.write(f"- Color: {INV_COLOR[pred_col]} (conf {conf_col:.1f}%)")
    st.write(f"- Size: {'B' if pred_size_idx==1 else 'S'} (conf {conf_size:.1f}%)")

    # recommend best bet among Number/Color/Size
    best_cat, best_val, best_conf = None, None, -1.0
    for cat, val, conf in [
        ('Number', pred_num, conf_num),
        ('Color', INV_COLOR[pred_col], conf_col),
        ('Size', 'B' if pred_size_idx==1 else 'S', conf_size)
    ]:
        if conf > best_conf:
            best_conf = conf; best_cat = cat; best_val = val

    if best_conf >= CONF_THRESHOLD:
        st.success(f"RECOMMENDED BET: {best_cat} -> {best_val} (confidence {best_conf:.1f}%)")
    else:
        st.warning("WAIT: model confidence below threshold. Keep collecting results until pattern emerges.")
else:
    need = max(0, WINDOW - len(st.session_state.history))
    st.info(f"Need {need} more rounds (history) to make reliable predictions.")

# ------------------------
# Confirm & Learn
# ------------------------
st.subheader("Confirm pending round and learn")
if st.session_state.pending:
    if st.button("Confirm & Learn"):
        new_round = st.session_state.pending
        # update priors from new round
        if len(st.session_state.history) >= 1:
            prev = st.session_state.history[-1]
            for p in range(NUM_PLACES):
                st.session_state.markov_num[p].update(prev['places'][p]['num'], new_round['places'][p]['num'])
        for p in range(NUM_PLACES):
            st.session_state.freq_num[p].update(new_round['places'][p]['num'])
            st.session_state.freq_col[p].update(new_round['places'][p]['color'])

        st.session_state.history.append(new_round)
        save_history()
        st.session_state.log.append({
            'added': round_to_display(new_round),
            'history_len': len(st.session_state.history)
        })
        st.session_state.pending = None
        st.success("Confirmed & learned — priors updated.")
        st.experimental_rerun()
    if st.button("Discard pending"):
        st.session_state.pending = None
        st.info("Pending discarded")

# ------------------------
# History & logs
# ------------------------
st.markdown("---")
st.subheader("History (last 200)")
if st.session_state.history:
    df = pd.DataFrame([{'Round': i+1,
                        'P1': f\"{r['places'][0]['num']}{INV_COLOR[r['places'][0]['color']]}{'B' if r['places'][0].get('size_observed', r['places'][0]['num']>=5) else 'S'}\",
                        'P2': f\"{r['places'][1]['num']}{INV_COLOR[r['places'][1]['color']]}{'B' if r['places'][1].get('size_observed', r['places'][1]['num']>=5) else 'S'}\",
                        'P3': f\"{r['places'][2]['num']}{INV_COLOR[r['places'][2]['color']]}{'B' if r['places'][2].get('size_observed', r['places'][2]['num']>=5) else 'S'}\"}
                       for i,r in enumerate(st.session_state.history)])
    st.dataframe(df.tail(200))
    buf = BytesIO(); df.to_excel(buf, index=False)
    st.download_button("⬇️ Download history (Excel)", data=buf.getvalue(), file_name="history_simple.xlsx")

if st.session_state.log:
    st.subheader("Log")
    st.dataframe(pd.DataFrame(st.session_state.log).tail(200))
    buf2 = BytesIO(); pd.DataFrame(st.session_state.log).to_excel(buf2, index=False)
    st.download_button("⬇️ Download log (Excel)", data=buf2.getvalue(), file_name="log_simple.xlsx")

st.caption("Notes: This is a practical, lightweight online predictor using Markov + frequency + recent pattern. It enforces number->size determinism and handles ambiguous colors (0/5) via color frequency priors.")
