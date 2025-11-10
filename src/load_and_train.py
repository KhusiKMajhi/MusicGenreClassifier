# src/load_and_train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# EDIT THIS to the CSV you prefer:
CSV_PATH = "data/features_3_sec.csv"   # or "data/features_30_sec.csv"
MODEL_OUT = "models/final_model.joblib"

def load_data(path):
    df = pd.read_csv(path)
    # auto detect label column
    for c in ['genre','label','class','label_genre']:
        if c in df.columns:
            label_col = c
            break
    else:
        label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
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
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    main()


#run it as
#python src\load_and_train.py
