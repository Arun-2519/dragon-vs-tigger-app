# ==============================
# DRAGON vs TIGER HYBRID AI SYSTEM
# LSTM + XGBOOST + PATTERN ENGINE + TELEGRAM BOT
# ==============================

import numpy as np
import pandas as pd
import joblib, os, time

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# 1️⃣ ENCODING
# ---------------------------------------------------
map_dict = {"D":0, "T":1, "DRAW":2}
reverse_map = {0:"D",1:"T",2:"DRAW"}

# ---------------------------------------------------
# 2️⃣ PREPARE SEQUENCES
# ---------------------------------------------------
def prepare_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# ---------------------------------------------------
# 3️⃣ TRAIN MODELS
# ---------------------------------------------------
def train_models(df):
    data = df["Result"].map(map_dict).values
    X, y = prepare_sequences(data, 10)

    # LSTM SHAPE
    X_lstm = X.reshape((X.shape[0], 10, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, shuffle=False)

    print("🎯 Training LSTM...")
    lstm = Sequential([
        LSTM(64, input_shape=(10,1)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    lstm_acc = lstm.evaluate(X_test, y_test, verbose=0)[1]

    print("🔥 Training XGBoost...")
    X_xgb = X.reshape(len(X), 10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_xgb, y, test_size=0.2, shuffle=False)

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
    xgb.fit(X_train2, y_train2)
    y_pred = xgb.predict(X_test2)
    xgb_acc = accuracy_score(y_test2, y_pred)

    # SAVE MODELS
    lstm.save("lstm_model.h5")
    joblib.dump(xgb, "xgb_model.pkl")

    print(f"✅ LSTM Accuracy: {lstm_acc:.2f}")
    print(f"✅ XGB Accuracy: {xgb_acc:.2f}")

    return lstm, xgb

# ---------------------------------------------------
# 4️⃣ HYBRID PREDICTOR
# ---------------------------------------------------
def hybrid_predict(last10):
    lstm = load_model("lstm_model.h5")
    xgb = joblib.load("xgb_model.pkl")

    arr = np.array(last10).reshape(1, 10)
    arr_lstm = arr.reshape(1,10,1)

    lstm_prob = lstm.predict(arr_lstm, verbose=0)[0]
    lstm_choice = np.argmax(lstm_prob)
    lstm_conf = lstm_prob[lstm_choice]

    xgb_choice = xgb.predict(arr)[0]

    if lstm_conf < 0.55:
        return xgb_choice, 0.55, "XGB (fallback)"
    
    return lstm_choice, lstm_conf, "LSTM"

# ---------------------------------------------------
# 5️⃣ AUTO RETRAIN FUNCTION (daily)
# ---------------------------------------------------
def auto_retrain(csv):
    print("♻ Auto retraining model...")
    df = pd.read_csv(csv)
    df["Result"] = df["Result"].str.upper().str.strip()
    train_models(df)
    print("✅ Retraining completed")

# ---------------------------------------------------
# 6️⃣ TELEGRAM BOT LIVE MODE
# ---------------------------------------------------
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

USER_SEQ = []

def start(update, context):
    update.message.reply_text("📊 Send latest result (D/T/DRAW). I will predict next.")

def get_signal(update, context):
    global USER_SEQ
    text = update.message.text.upper().strip()

    if text not in map_dict:
        return update.message.reply_text("⚠ Send only D / T / DRAW")

    USER_SEQ.append(map_dict[text])

    if len(USER_SEQ) < 10:
        return update.message.reply_text(f"Need {10-len(USER_SEQ)} more inputs...")

    last10 = USER_SEQ[-10:]
    pred, conf, source = hybrid_predict(last10)
    update.message.reply_text(
        f"🧠 Prediction: {reverse_map[pred]}\n"
        f"📈 Confidence: {conf:.2f}\n"
        f"🔗 Model: {source}"
    )

def run_telegram_bot():
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text, get_signal))

    updater.start_polling()
    updater.idle()

# ---------------------------------------------------
# 7️⃣ MAIN MENU
# ---------------------------------------------------
print("""
✅ Dragon vs Tiger AI System Loaded
1) Train model from CSV
2) Predict manually in console
3) Start Telegram Bot
4) Auto Retrain
""")

choice = input("Select option: ")

if choice == "1":
    file = input("Enter CSV file name: ")
    df = pd.read_csv(file)
    df["Result"] = df["Result"].astype(str).str.upper().str.strip()
    train_models(df)

elif choice == "2":
    print("Enter results one by one (D/T/DRAW)")
    buffer = []
    while True:
        val = input("Result: ").upper()
        if val not in map_dict: continue
        buffer.append(map_dict[val])
        if len(buffer) >= 10:
            pred, c, src = hybrid_predict(buffer[-10:])
            print(f"Next Prediction: {reverse_map[pred]} | {c:.2f} | {src}")

elif choice == "3":
    run_telegram_bot()

elif choice == "4":
    file = input("Enter CSV for auto retrain: ")
    auto_retrain(file)
