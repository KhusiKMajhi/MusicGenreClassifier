# src/recommender.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, features_csv="data/features_3_sec.csv"):
        self.df = pd.read_csv(features_csv)
        # detect label col
        self.label_col = next((c for c in self.df.columns if 'genre' in c.lower()), self.df.columns[-1])
        self.X = self.df.drop(columns=[self.label_col])
        for c in ['path','filename','file','files','file_name']:
            if c in self.X.columns:
                self.X = self.X.drop(columns=[c])
        # normalize
        self.norm = (self.X - self.X.mean()) / (self.X.std() + 1e-9)

    def recommend_by_index(self, idx, top_k=5):
        sims = cosine_similarity([self.norm.iloc[idx]], self.norm)[0]
        order = np.argsort(-sims)
        recs = [i for i in order if i != idx][:top_k]
        return self.df.iloc[recs][['path', self.label_col]].to_dict(orient='records')

if __name__ == "__main__":
    r = Recommender()
    print(r.recommend_by_index(0, top_k=5))



#run it as
#python src\recommender.py
