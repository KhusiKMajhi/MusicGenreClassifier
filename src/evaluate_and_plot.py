# src/evaluate_and_plot.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_PATH = "models/final_model.joblib"
CSV_PATH = "data/features_3_sec.csv"  # change to 30s if desired

def load_df(path):
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()), df.columns[-1])
    X = df.drop(columns=[label_col])
    for c in ['path','filename','file','files','file_name']:
        if c in X.columns:
            X = X.drop(columns=[c])
    return df, X, df[label_col].values, label_col

def plot_cm(y_true, y_pred, out="reports/confusion_matrix.png"):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out)
    print("Saved confusion matrix to", out)
    plt.show()

def main():
    model = joblib.load(MODEL_PATH)
    df, X, y_true, label_col = load_df(CSV_PATH)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    plot_cm(y_true, y_pred)

if __name__ == "__main__":
    main()


#run it as
#python src\evaluate_and_plot.py
