# MusicGenreClassifier.py

import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
from pydub import AudioSegment
import time

st.set_page_config(page_title="Trasetone", layout="wide")

st.title("🎶 Trasetone: Genre & Mood Classifier")

# Sidebar – Upload audio
st.sidebar.header("📁 Upload Audio")
audio_file = st.sidebar.file_uploader("Choose an MP3 or WAV file", type=["mp3", "wav"])

# Preprocessing
def preprocess_audio(file):
    if file.type == 'audio/mp3':
        audio = AudioSegment.from_mp3(file)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        y, sr = librosa.load(buffer, sr=22050, mono=True)
    else:
        y, sr = librosa.load(file, sr=22050, mono=True)
    y = librosa.util.normalize(y)
    return y, sr

# Feature extraction
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        zcr.mean(),
        tempo
    ])
    return feature_vector, mfcc

# Visualize MFCC
def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', cmap='magma')
    plt.colorbar()
    plt.title('MFCC Features')
    st.pyplot(fig)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("genre_mood_classifier.pkl")

# Label decoder (optional)
def decode_label(index):
    try:
        encoder = joblib.load("genre_mood_label_encoder.pkl")
        return encoder.inverse_transform([index])[0]
    except:
        return f"Label #{index}"

# Recommender
def recommend_songs(label):
    return {
        "Pop-Happy": ["Blinding Lights", "Levitating", "Uptown Funk"],
        "Rock-Sad": ["Nothing Else Matters", "Creep", "Tears in Heaven"],
        "Jazz-Calm": ["Take Five", "Blue in Green", "Autumn Leaves"],
        "EDM-Energetic": ["Levels", "Animals", "Titanium"]
    }.get(label, ["No recommendations available."])

# Main App Logic
if audio_file:
    st.audio(audio_file, format="audio/wav")
    y, sr = preprocess_audio(audio_file)

    st.subheader("📊 Audio Visualization")
    _, mfcc = extract_features(y, sr)
    plot_mfcc(mfcc)

    st.subheader("🎧 Prediction")
    start = time.time()
    features, _ = extract_features(y, sr)
    model = load_model()
    prediction_encoded = model.predict([features])[0]
    end = time.time()

    # Decode label
    prediction_label = decode_label(prediction_encoded)
    st.success(f"🎼 Predicted Genre & Mood: **{prediction_label}**")
    st.caption(f"🕒 Processed in {end - start:.2f} seconds")

    st.subheader("🎵 Recommendations")
    for song in recommend_songs(prediction_label):
        st.markdown(f"- {song}")

st.markdown("---")
st.caption("Developed by Group 2 – Trasetone · 2025")
