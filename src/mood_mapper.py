# src/mood_mapper.py
import pandas as pd

def map_mood(row):
    tempo = float(row.get('tempo', 0))
    energy = float(row.get('rmse', row.get('rms_mean', 0)))
    if tempo >= 120 and energy > 0.04:
        return "Energetic"
    if 90 <= tempo < 120 and energy > 0.03:
        return "Happy"
    if tempo < 90 and energy < 0.03:
        return "Calm"
    if tempo < 80 and energy < 0.02:
        return "Sad"
    return "Neutral"

def main(csv_in="data/features_3_sec.csv", csv_out="data/features_with_moods.csv"):
    df = pd.read_csv(csv_in)
    df['mood'] = df.apply(map_mood, axis=1)
    df.to_csv(csv_out, index=False)
    print("Saved with moods to", csv_out)

if __name__ == "__main__":
    main()


#run it as
#python src\mood_mapper.py
