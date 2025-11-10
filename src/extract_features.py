# src/extract_features.py
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join("data", "genres_original")  # adjust if needed
OUT_CSV = os.path.join("data", "features_from_audio.csv")
SR = 22050
DURATION = 30.0  # seconds to load each file

def extract_one(path, sr=SR, duration=DURATION):
    try:
        y, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    except Exception as e:
        print("Failed to load:", path, e)
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rmse = librosa.feature.rms(y).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features = np.hstack([mfcc_mean, mfcc_var, chroma_mean, spec_cent, zcr, rmse, tempo])
    return features

def main():
    rows = []
    genres = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    genres.sort()
    for genre in genres:
        folder = os.path.join(DATA_DIR, genre)
        for fname in tqdm(os.listdir(folder), desc=genre):
            if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                continue
            path = os.path.join(folder, fname)
            feat = extract_one(path)
            if feat is None:
                continue
            row = feat.tolist()
            row.append(genre)
            row.append(path)
            rows.append(row)

    # columns
    cols = []
    for i in range(13):
        cols.append(f"mfcc_mean_{i+1}")
    for i in range(13):
        cols.append(f"mfcc_var_{i+1}")
    for i in range(12):
        cols.append(f"chroma_{i+1}")
    cols += ["spec_centroid", "zcr", "rmse", "tempo", "genre", "path"]
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved features to", OUT_CSV)

if __name__ == "__main__":
    main()


#run it as
#python src\extract_features.py
