from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    X = df.drop(["filename", "length", "label"], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, le

#To RUN it in terminal
#python src/preprocess.py