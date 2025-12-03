genre_to_mood = {
    'pop': 'happy',
    'rock': 'energetic',
    'metal': 'aggressive',
    'blues': 'sad',
    'country': 'calm',
    'disco': 'happy',
    'hiphop': 'excited',
    'classical': 'peaceful',
    'reggae': 'chill',
    'jazz': 'calm'
}

def predict_genre_and_mood(model, scaler, features, label_encoder):
    import pandas as pd
    features_df = pd.DataFrame([features], columns=scaler.feature_names_in_)
    scaled = scaler.transform(features_df)
    genre_index = model.predict(scaled)[0]
    genre = label_encoder.inverse_transform([genre_index])[0]
    mood = genre_to_mood.get(genre, "unknown")
    return genre, mood

#To RUN it in terminal
#python src/mood_mapping.py