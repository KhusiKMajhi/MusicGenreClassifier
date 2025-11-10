# app/streamlit_app.py
import streamlit as st
import joblib
import tempfile
import pandas as pd
import numpy as np
import librosa
from src.extract_features import extract_one if False else None  # placeholder: we will implement feature extraction in-line

MODEL_PATH = "models/final_model.joblib"
st.title("Music Genre Classifier Demo")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.warning("Model not found. Run training first.")
    model = None

uploaded = st.file_uploader("Upload audio (wav/mp3)", type=["wav","mp3","flac","m4a"])
if uploaded and model:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.write(uploaded.read())
    tf.flush()
    # Extract features inline: keep consistent with training features (13 mfcc means, etc.)
    y, sr = librosa.load(tf.name, sr=22050, mono=True, duration=30.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rmse = librosa.feature.rms(y).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feat = np.hstack([mfcc_mean, mfcc_var, chroma, spec_cent, zcr, rmse, tempo])
    # Build DataFrame with same column order as training (attempt)
    # Construct columns like in extract_features.py
    cols = []
    for i in range(13):
        cols.append(f"mfcc_mean_{i+1}")
    for i in range(13):
        cols.append(f"mfcc_var_{i+1}")
    for i in range(12):
        cols.append(f"chroma_{i+1}")
    cols += ["spec_centroid","zcr","rmse","tempo"]
    X = pd.DataFrame([feat], columns=cols)
    pred = model.predict(X)[0]
    st.success(f"Predicted genre: {pred}")

#run it as
#streamlit run app\streamlit_app.py
