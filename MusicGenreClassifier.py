import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
from pydub import AudioSegment

# Page Configuration
st.set_page_config(page_title="Trasetone: Genre & Mood Classifier", layout="centered")

st.title("🎵 Trasetone: Genre & Mood Classification System")
st.markdown("Upload a short audio clip to classify its genre and mood, and get smart song recommendations!")

# --- Upload Audio ---
audio_file = st.file_uploader("📂 Upload an MP3 or WAV file", type=["mp3", "wav"])

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

# --- Feature Extraction ---
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]  # ✅ Fixed: use keyword argument

    feature_vector = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        zcr.mean(),
        tempo
    ])
    return feature_vector, mfcc

# --- MFCC Visualization ---
def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC (Mel Frequency Cepstral Coefficients)')
    plt.tight_layout()
    st.pyplot(fig)

# --- Load Classifier Model ---
@st.cache_resource
def load_model():
    return joblib.load("genre_mood_classifier.pkl")

# --- Recommendation System ---
def recommend_songs(prediction):
    recommendations = {
        "Pop-Happy": ["Blinding Lights", "Levitating", "Uptown Funk"],
        "Rock-Sad": ["Nothing Else Matters", "Tears in Heaven", "Creep"],
        "Jazz-Calm": ["Blue in Green", "Take Five", "Autumn Leaves"],
        "EDM-Energetic": ["Animals", "Levels", "Wake Me Up"],
    }
    return recommendations.get(prediction, ["No recommendations available."])

# --- Main Logic ---
if audio_file:
    st.audio(audio_file, format='audio/wav')
    y, sr = preprocess_audio(audio_file)

    st.subheader("🎼 Audio Feature Visualization")
    _, mfcc = extract_features(y, sr)
    plot_mfcc(mfcc)

    st.subheader("🎧 Genre & Mood Prediction")
    features, _ = extract_features(y, sr)
    model = load_model()
    prediction = model.predict([features])[0]
    st.success(f"🔍 Detected Genre & Mood: **{prediction}**")

    st.subheader("🔁 Smart Song Recommendations")
    for song in recommend_songs(prediction):
        st.markdown(f"- 🎵 {song}")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Group 2 - CPDSPG1L · Trasetone Project · 2025")
