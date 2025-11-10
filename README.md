# Music Genre Classification

Simple ML project to classify music genres using audio features (MFCC, chroma, spectral features).

## Run locally

1. Create and activate venv
2. pip install -r requirements.txt
3. Place dataset under data/
4. python src/load_and_train.py
5. python src/evaluate_and_plot.py
6. streamlit run app/streamlit_app.py

## Files

- src/load_and_train.py : train and save model
- src/evaluate_and_plot.py : evaluate and plot confusion matrix
- src/test_model_improved.py : test script printing predicted vs actual
- app/streamlit_app.py : simple demo

# 🎵 Music Genre Classification

This project classifies songs into different genres using Machine Learning techniques and audio features.

## 📘 Overview

This model uses extracted features (MFCC, spectral contrast, etc.) to train a Random Forest Classifier that predicts the genre of a song segment.

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Librosa
- Matplotlib, Seaborn

## 🚀 How to Run

1. Clone this repository
2. Create a virtual environment:

🧠 Model file (final_model.joblib) is stored locally in the /models folder.
To run predictions, train the model first or place your trained model file in this folder.

#REMEMBER
STEP1: activate venv by the command
.\venv\Scripts\activate
