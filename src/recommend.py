from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_songs(df, scaler, mood, top_n=5):
    df["mood"] = df["label"].map({
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
    })

    mood_df = df[df["mood"] == mood]
    if len(mood_df) == 0:
        return []

    sample = mood_df.sample(1)
    sample_features = scaler.transform(sample.drop(['filename','length','label','mood'], axis=1))

    features_scaled = scaler.transform(df.drop(['filename','length','label','mood'], axis=1))
    sim_scores = cosine_similarity(sample_features, features_scaled)[0]

    top_indices = np.argsort(sim_scores)[-top_n:]
    return df.iloc[top_indices][["filename", "label", "mood"]]

#To RUN it in terminal
#python src/recommend.py