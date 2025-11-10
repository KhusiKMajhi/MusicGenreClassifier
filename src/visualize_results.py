import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# Load data
CSV_PATH = "data/features_30_sec.csv"   # or your correct CSV file
df = pd.read_csv(CSV_PATH)
label_col = [c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()]
label_col = label_col[0] if label_col else df.columns[-1]

X = df.drop(columns=[label_col])
for c in ['path','filename','file','files','file_name']:
    if c in X.columns:
        X = X.drop(columns=[c])
y = df[label_col]

# Load model
model = joblib.load("models/final_model.joblib")

# Split data (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Predict
preds = model.predict(X_test)

# Classification report
print("\n🎯 Classification Report:\n")
print(classification_report(y_test, preds))

# Confusion Matrix
cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🎵 Music Genre Classification Confusion Matrix")
plt.tight_layout()
plt.show()

#run it as
#python src/visualize_results.py
