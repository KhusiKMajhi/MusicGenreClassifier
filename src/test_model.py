#Using the trained model for prediction 🎧

# src/test_model_improved.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------- CONFIGURE THIS ----------
MODEL_PATH = "models/final_model.joblib"
CSV_PATH   = "data/features_3_sec.csv"   # <-- use the same CSV you trained on (3_sec or 30_sec)
SAMPLE_N   = 20                          # how many random examples to show
RANDOM_SEED = 42
# -----------------------------------

def load_df(path):
    df = pd.read_csv(path)
    # detect label column
    label_col = next((c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()), None)
    if not label_col:
        label_col = df.columns[-1]
    return df, label_col

def preprocess_X(df, label_col):
    X = df.drop(columns=[label_col])
    # drop typical non-feature columns if present
    for c in ['path','filename','file','files','file_name']:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X

def main():
    # load model & data
    model = joblib.load(MODEL_PATH)
    df, label_col = load_df(CSV_PATH)
    X = preprocess_X(df, label_col)
    y = df[label_col].values

    # Full-dataset evaluation
    y_pred_all = model.predict(X)
    acc = accuracy_score(y, y_pred_all)
    print(f"\n🎯 Overall accuracy on {CSV_PATH}: {acc:.3f}\n")

    # Print classification report (text)
    print("📋 Classification report:\n")
    print(classification_report(y, y_pred_all, digits=3))

    # Confusion matrix (text)
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred_all, labels=labels)
    print("🧾 Confusion matrix (rows=true, cols=pred):\n")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # Show SAMPLE_N random samples: predicted vs actual
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False)
    preds = model.predict(X.iloc[idx])
    actuals = y[idx]

    # Print arrays in requested format
    print("\n🎵 Predicted Genres:", np.array2string(np.array(preds), separator=" ", max_line_width=200))
    print("✅ Actual Genres:   ", np.array2string(np.array(actuals), separator=" ", max_line_width=200))

    # Also show a clear side-by-side list
    print("\n🔎 Sample (index, predicted, actual):")
    for i, pred, act in zip(idx, preds, actuals):
        path = df.get('path', pd.Series([None]*len(df))).iloc[i]
        print(f"{i:5d}  {pred:12s}   {act:12s}   {path if pd.notna(path) else ''}")

if __name__ == "__main__":
    main()

#run the code in terminal using
#python src/test_model.py
