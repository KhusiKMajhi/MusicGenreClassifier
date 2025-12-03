
import streamlit as st
import pandas as pd

from src.preprocess import preprocess
from src.train_model import train_model
from src.mood_mapping import predict_genre_and_mood
from src.recommend import recommend_songs

st.title("🎵 Music Genre & Mood Recommendation System")

df = pd.read_csv("data/features_3_sec.csv")
X_train, X_test, y_train, y_test, scaler, le = preprocess(df)
model = train_model(X_train, y_train, X_test, y_test, le)

st.subheader("🔍 Predict Mood of a Song (Manual Input)")
sample_features = df.drop(['filename','length','label'], axis=1).iloc[0].values
genre, mood = predict_genre_and_mood(model, scaler, sample_features, le)
st.write("Predicted Genre:", genre)
st.write("Predicted Mood:", mood)

st.subheader("🎶 Mood-Based Song Recommendations")
user_mood = st.selectbox("Select Mood:", ["happy", "sad", "calm", "excited", "energetic", "chill", "peaceful"])
if st.button("Recommend"):
    results = recommend_songs(df, scaler, user_mood)
    st.write(results)

#to run use
#streamlit run app.py
