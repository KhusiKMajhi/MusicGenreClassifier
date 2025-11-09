# src/load_and_train.py
#This trains a RandomForest, prints metrics and saves the model.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_PATH = "data/features_30_sec.csv"   # or "data/features_3_sec.csv" for quick test

def load_data(path):
    df = pd.read_csv(path)
    # try to identify label column
    for c in ['genre','label','class','label_genre']:
        if c in df.columns:
            label_col = c
            break
    else:
        label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
    # drop non-feature columns if present
    for c in ['path','filename','file','files','file_name']:
        if c in X.columns:
            X = X.drop(columns=[c])
    y = df[label_col]
    return X, y

def main():
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
    print("Training...")
    pipe.fit(X_train, y_train)
    print("Evaluating...")
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/final_model.joblib")
    print("Saved model to models/final_model.joblib")

if __name__ == "__main__":
    main()

#run it using this in terminal to execute it
#python src\load_and_train.py
