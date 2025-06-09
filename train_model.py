# train_model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Simulate 100 samples
n_samples = 100
n_features = 13 * 2 + 12 * 2 + 1 + 1  # MFCC mean/std, Chroma mean/std, ZCR, Tempo

# Generate random features
X = np.random.rand(n_samples, n_features)

# Generate random labels like Pop-Happy, Rock-Sad, etc.
genres = ['Pop', 'Rock', 'Jazz', 'EDM']
moods = ['Happy', 'Sad', 'Calm', 'Energetic']
y = [f"{g}-{m}" for g, m in zip(np.random.choice(genres, n_samples), np.random.choice(moods, n_samples))]

# Encode text labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder (optional)
joblib.dump(label_encoder, "genre_mood_label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.2f}")

# Save the trained model
joblib.dump(model, "genre_mood_classifier.pkl")
print("✅ Model saved as genre_mood_classifier.pkl")
