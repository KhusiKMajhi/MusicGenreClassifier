import pandas as pd

'''
# Use the same path that worked before
p = "data/features_3_sec.csv"
df = pd.read_csv(p)

# Detect possible label column
possible_labels = [c for c in df.columns if 'genre' in c.lower() or 'label' in c.lower()]
label_col = possible_labels[0] if possible_labels else df.columns[-1]

print("✅ Using label column:", label_col)
print("\n📊 Class distribution:")
print(df[label_col].value_counts())

print("\n🎧 Sample data:")
print(df[[label_col] + df.columns[:5].tolist()].head())
'''


#Count records per genre (class distribution) — helps check balance:
'''
p = "data/features_30_sec.csv"
df = pd.read_csv(p)
# find label column
for c in ['genre','label','class','label_genre']:
    if c in df.columns:
        lab = c
        break
else:
    lab = df.columns[-1]
print("Label column:", lab)
print(df[lab].value_counts())
'''

#Check for missing values (quick):

p = "data/features_30_sec.csv"
df = pd.read_csv(p)
print("Missing per column (non-zero only):")
print(df.isna().sum()[lambda s: s>0])

#all code run successfully