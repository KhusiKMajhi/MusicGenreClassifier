# src/evaluate_and_plot.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_PATH = "models/final_model.joblib"
CSV_PATH = "data/features_3_sec.csv"   # change to features_30_sec.csv if needed

def load_data(path):
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()), df.columns[-1])
    X = df.drop(columns=[label_col])
    for c in ['path','filename','file','files','file_name']:
        if c in X.columns:
            X = X.drop(columns=[c])
    return df, X, df[label_col].values, label_col

def plot_confusion(y_true, y_pred, out_file="reports/confusion_matrix.png"):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved confusion matrix to {out_file}")
    plt.show()

def main():
    print("Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    df, X, y_true, label_col = load_data(CSV_PATH)
    print("Dataset:", CSV_PATH, "| Label column:", label_col)
    # predict
    y_pred = model.predict(X)
    # metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc:.3f}\n")
    print("Classification report:\n")
    print(classification_report(y_true, y_pred, digits=3))
    # confusion
    plot_confusion(y_true, y_pred)

if __name__ == "__main__":
    main()


#python src/evaluate_and_plot.py
