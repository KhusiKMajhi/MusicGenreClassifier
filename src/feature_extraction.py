import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
    
    features = np.hstack([mfccs_mean, chroma_mean, spec_contrast_mean])
    return features

#to run use
#python src/feature_extraction.py