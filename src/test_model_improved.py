# src/test_model_improved.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "models/final_model.joblib"
CSV_PATH = "data/features_3_sec.csv"  # or features_30_sec.csv
SAMPLE_N = 20
RANDOM_SEED = 42

def load_df(path):
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()), df.columns[-1])
    return df, label_col

def preprocess_X(df, label_col):
    X = df.drop(columns=[label_col])
    for c in ['path','filename','file','files','file_name']:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X

def main():
    model = joblib.load(MODEL_PATH)
    df, label_col = load_df(CSV_PATH)
    X = preprocess_X(df, label_col)
    y = df[label_col].values

    y_pred_all = model.predict(X)
    acc = accuracy_score(y, y_pred_all)
    print(f"\n🎯 Overall accuracy on {CSV_PATH}: {acc:.3f}\n")
    print(classification_report(y, y_pred_all, digits=3))

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False)
    preds = model.predict(X.iloc[idx])
    actuals = y[idx]
    # print arrays similar to your requested style
    print("\n🎵 Predicted Genres:", np.array2string(np.array(preds), separator=" ", max_line_width=200))
    print("✅ Actual Genres:   ", np.array2string(np.array(actuals), separator=" ", max_line_width=200))
    # side by side
    print("\n🔎 Sample (index | predicted | actual | path):")
    for i, pred, act in zip(idx, preds, actuals):
        path = df.get('path', pd.Series([None]*len(df))).iloc[i]
        print(f"{i:5d}  {pred:12s}   {act:12s}   {path if pd.notna(path) else ''}")

if __name__ == "__main__":
    main()


#run it as
#python src\test_model_improved.py
