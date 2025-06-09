import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
import time
from pydub import AudioSegment

st.set_page_config(page_title="Trasetone", layout="wide")
st.title("🎶 Trasetone: Genre & Mood Classifier")

# Sidebar upload
st.sidebar.header("📁 Upload Audio")
audio_file = st.sidebar.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])

# --- Preprocessing ---
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

# --- Feature extraction ---
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
    return feature_vector, mfcc, chroma

# --- Visualizations ---
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    ax.set_title("📈 Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_chroma(chroma):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
    ax.set_title("🎹 Chroma Spectrogram")
    st.pyplot(fig)

def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mfcc, x_axis='time', cmap='magma')
    plt.colorbar()
    ax.set_title("📊 MFCC")
    st.pyplot(fig)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("genre_mood_classifier.pkl")

# --- Decode Label ---
def decode_label(index):
    try:
        encoder = joblib.load("genre_mood_label_encoder.pkl")
        return encoder.inverse_transform([index])[0]
    except:
        return f"Label #{index}"

# --- Recommender ---
def recommend_songs(label):
    return {
        "Pop-Happy": ["Blinding Lights", "Levitating", "Uptown Funk"],
        "Rock-Sad": ["Nothing Else Matters", "Creep", "Tears in Heaven"],
        "Jazz-Calm": ["Take Five", "Blue in Green", "Autumn Leaves"],
        "EDM-Energetic": ["Levels", "Animals", "Titanium"]
    }.get(label, ["No recommendations available."])

# --- App Logic ---
if audio_file:
    st.audio(audio_file, format="audio/wav")
    y, sr = preprocess_audio(audio_file)

    st.subheader("🎼 Audio Visualizations")
    feature_vec, mfcc, chroma = extract_features(y, sr)
    plot_waveform(y, sr)
    plot_chroma(chroma)
    plot_mfcc(mfcc)

    st.subheader("🎧 Genre & Mood Prediction")
    start = time.time()
    model = load_model()
    prediction_encoded = model.predict([feature_vec])[0]
    end = time.time()

    prediction_label = decode_label(prediction_encoded)
    st.success(f"🎼 Detected: **{prediction_label}**")
    st.caption(f"🕒 Inference Time: {end - start:.2f} seconds")

    st.subheader("🎵 Recommended Songs")
    for song in recommend_songs(prediction_label):
        st.markdown(f"- {song}")

# Footer
st.markdown("---")
st.caption("Developed by Group 2 – Trasetone Project – 2025")
