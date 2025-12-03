from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_songs(df, scaler, mood, top_n=5):
    # Map genres to moods
    mood_map = {
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

    # Add mood column
    df["mood"] = df["label"].map(mood_map)

    # Filter songs with requested mood
    mood_df = df[df["mood"] == mood]

    if len(mood_df) == 0:
        print("⚠️ No songs found for this mood!")
        return []

    # Pick one sample song
    sample = mood_df.sample(1)

    feature_columns = df.columns.drop(["filename","length","label","mood"])
    
    sample_features = scaler.transform(sample[feature_columns])
    all_features = scaler.transform(df[feature_columns])

    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(sample_features, all_features)[0]

    top_indices = sim_scores.argsort()[-top_n:]

    return df.iloc[top_indices][["filename", "label", "mood"]]


#To RUN it in terminal
#python src/recommend.py