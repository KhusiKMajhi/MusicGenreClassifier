import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/features_30_sec.csv")

# Drop unnecessary columns
X = data.drop(columns=['label', 'filename'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Training complete!")
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
joblib.dump(model, "models/final_model.joblib")
print("💾 Model saved to models/final_model.joblib")

#run it as
#python src/train_model.py