import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, html, unicodedata
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json


# Load tokenizer properly
with open("tokenizer.json") as f:
    data = f.read()   # read raw string
tok = tokenizer_from_json(data)

# ========================
# 1. Load Artifacts
# ========================
@st.cache_resource
def load_artifacts():
    model = load_model("model_bilstm.keras")
    tok = joblib.load("tokenizer.joblib")
    meta = joblib.load("meta.joblib")
    return model, tok, meta

model, tok, meta = load_artifacts()

# ========================
# 2. Text Cleaning
# ========================
def normalize_text(s):
    s = str(s).lower()
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"http\S+|www\S+", " url ", s)
    s = re.sub(r"@\w+", " user ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def predict_text(text):
    seq = pad_sequences(tok.texts_to_sequences([normalize_text(text)]),
                        maxlen=meta["max_len"], padding="post")
    prob = model.predict(seq, verbose=0)[0][0]
    return prob, int(prob >= meta["threshold"])

# ========================
# 3. Streamlit UI
# ========================
st.title("🛡️ Comment Toxicity Detector")
st.write("Detect whether a comment is toxic using a trained BiLSTM model.")

# ---- Single text prediction
st.subheader("Single Comment Prediction")
user_input = st.text_area("Enter a comment:")

if st.button("Predict"):
    prob, pred = predict_text(user_input)
    st.write(f"**Toxic Probability:** {prob:.2f}")
    st.success("Non-toxic ✅" if pred==0 else "🚨 Toxic")

# ---- CSV upload
st.subheader("Bulk Prediction from CSV")
file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
if file:
    df = pd.read_csv(file)
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        seqs = pad_sequences(tok.texts_to_sequences(df["text"].astype(str).apply(normalize_text)),
                             maxlen=meta["max_len"], padding="post")
        probs = model.predict(seqs, verbose=0).ravel()
        df["toxic_prob"] = probs
        df["prediction"] = (probs >= meta["threshold"]).astype(int)
        st.write(df.head())
        st.download_button("Download Predictions", df.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")

