#testing the Dataset CSV file

import pandas as pd

'''
p = "data/features_3_sec.csv"
df = pd.read_csv(p)

print("✅ FILE LOADED:", p)
print("📊 Shape (rows, columns):", df.shape)
print("🧩 First 20 columns:", df.columns.tolist()[:20])'''

#Check first 5 rows to see sample data
'''
p = "data/features_30_sec.csv"
df = pd.read_csv(p)
print(df.head().to_string())'''

#Find which column is the label (genre)
#run this to list candidate label names and unique values.
'''
p = "data/features_30_sec.csv"
df = pd.read_csv(p)
# possible label names
candidates = [c for c in df.columns if c.lower() in ('genre','label','class','label_genre')]
print("Candidate label columns:", candidates)
# If none, show last column name and unique values (common in GTZAN)
if not candidates:
    last = df.columns[-1]
    print("Using last column as label:", last)
    print("Unique labels (sample):", df[last].unique()[:20])
else:
    lab = candidates[0]
    print("Label column chosen:", lab)
    print("Unique labels (sample):", df[lab].unique()[:20])

'''


#all code run successfully.