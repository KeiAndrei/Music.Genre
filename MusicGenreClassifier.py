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

# Title
st.title("🎶 Trasetone")

# Upload Section
st.sidebar.header("📁 File Upload and Audio Playback")
audio_file = st.sidebar.file_uploader("Choose a .mp3 or .wav file to upload", type=["mp3", "wav"])

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

# Feature Extraction
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
    return feature_vector, mfcc, chroma, y

# Visualization
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, color='blue')
    ax.set_title('Waveform')
    st.pyplot(fig)

def plot_mfcc(mfcc):
    fig, ax = plt.subplots()
    librosa.display.specshow(mfcc, x_axis='time', cmap='inferno')
    plt.colorbar()
    ax.set_title('MFCC')
    st.pyplot(fig)

def plot_chroma(chroma):
    fig, ax = plt.subplots()
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='plasma')
    plt.colorbar()
    ax.set_title('Chroma')
    st.pyplot(fig)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("genre_mood_classifier.pkl")

# Recommend songs
def recommend_songs(genre, mood):
    return [
        ("Song Title 1", "Artist 1"),
        ("Song Title 2", "Artist 2"),
        ("Song Title 3", "Artist 3"),
        ("Song Title 4", "Artist 4")
    ]

# Main Display
if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Start timer
    start_time = time.time()

    y, sr = preprocess_audio(audio_file)
    features, mfcc, chroma, y_wave = extract_features(y, sr)

    model = load_model()
    pred = model.predict([features])[0]
    confidence = model.predict_proba([features]).max() * 100
    genre, mood = pred.split("-") if "-" in pred else (pred, "Unknown")

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    # Layout
    left, right = st.columns([2, 1])

    with left:
        st.subheader("🎼 Genre and Mood Classification Output")
        st.write(f"**Genre:** {genre}")
        st.write(f"**Mood:** {mood}")

        st.subheader("📊 Audio Feature Visualization")
        plot_waveform(y_wave, sr)
        plot_chroma(chroma)
        plot_mfcc(mfcc)

    with right:
        st.subheader("🎧 Smart Song Recommendations")
        for i, (title, artist) in enumerate(recommend_songs(genre, mood), 1):
            st.write(f"{i}. **{title}** — *{artist}*")

        st.subheader("📈 Model Performance Feedback")
        st.write(f"**Processing Time:** {processing_time}s")
        st.write(f"**Prediction Confidence:** {confidence:.1f}%")

        if st.button("🔍 View Raw Features"):
            st.json({ "MFCC shape": str(mfcc.shape), "Chroma shape": str(chroma.shape) })

