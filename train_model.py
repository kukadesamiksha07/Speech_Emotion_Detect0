import os
import glob
import time
import numpy as np
import joblib
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

DATASET_DIR = r"E:\ML project\Datasets"
MODEL_SAVE_PATH = r"E:\ML project\VoxAura\speech_emotion_model.pkl"
CACHE_PATH = r"E:\ML project\VoxAura\robust_features_cache.npz"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features_from_audio(y, sample_rate=22050):
    """
    Extracts 236 audio features after speech-silence trimming and peak normalization.
    Ensures identical feature geometry between training and live microphone inference.
    """
    # 1. Trim leading and trailing silence to isolate active speech
    if len(y) > 2048:
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) >= 2048:
            y = y_trimmed

    # 2. Peak normalization (eliminates microphone gain / recording distance discrepancies)
    y = librosa.util.normalize(y)

    features = []

    # MFCC (40 coefficients -> mean & std = 80)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma STFT (12 bins -> mean & std = 24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Mel Spectrogram (64 bins converted to dB -> mean & std = 128)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # Zero-Crossing Rate (1 row -> mean & std = 2)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.extend(np.mean(zcr, axis=1))
    features.extend(np.std(zcr, axis=1))

    # Spectral Centroid (1 row -> mean & std = 2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
    features.extend(np.mean(centroid, axis=1))
    features.extend(np.std(centroid, axis=1))

    return np.array(features, dtype=np.float32)

def load_or_extract_features():
    if os.path.exists(CACHE_PATH):
        print(f"Loading robust cached features from {CACHE_PATH}...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return data["X"], data["y"], data["files"]

    search_path = os.path.join(DATASET_DIR, "Actor_*", "*.wav")
    audio_files = sorted(glob.glob(search_path))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {DATASET_DIR}")

    print(f"Extracting robust features for {len(audio_files)} RAVDESS files...")

    def process_file(file_path):
        filename = os.path.basename(file_path)
        parts = filename.split(".")[0].split("-")
        if len(parts) < 7:
            return None
        emotion = EMOTION_MAP.get(parts[2])
        if not emotion:
            return None
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        feat = extract_features_from_audio(y, sr)
        return feat, emotion, file_path

    t0 = time.time()
    results = Parallel(n_jobs=8, verbose=1)(delayed(process_file)(f) for f in audio_files)
    print(f"Feature extraction completed in {time.time() - t0:.2f} seconds.")

    valid = [r for r in results if r is not None]
    X = np.array([r[0] for r in valid], dtype=np.float32)
    y = np.array([r[1] for r in valid])
    files = np.array([r[2] for r in valid])

    print(f"Saving extracted features to {CACHE_PATH}...")
    np.savez_compressed(CACHE_PATH, X=X, y=y, files=files)
    return X, y, files

def train_and_save():
    X, y, files = load_or_extract_features()
    print(f"Total dataset: {len(X)} samples, {X.shape[1]} features.")

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # High-generalization ensemble combining Neural Network and Tree ensembles
    ensemble = VotingClassifier(
        estimators=[
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=600,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=25
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ],
        voting="soft"
    )

    print("\nTraining candidate ensemble on train split...")
    ensemble.fit(X_train_sc, y_train)

    train_acc = accuracy_score(y_train, ensemble.predict(X_train_sc))
    test_pred = ensemble.predict(X_test_sc)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")

    print("\n--- Detailed Test Classification Report ---")
    print(classification_report(y_test, test_pred, digits=4))

    labels = sorted(list(set(y)))
    print("Confusion Matrix:")
    print("Labels:", labels)
    print(confusion_matrix(y_test, test_pred, labels=labels))

    # Train production model on all 1,440 samples
    print("\nFitting final production pipeline on all 1,440 samples...")
    prod_scaler = StandardScaler()
    prod_classifier = VotingClassifier(
        estimators=[
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=600,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=25
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ],
        voting="soft"
    )

    prod_pipeline = Pipeline([
        ("scaler", prod_scaler),
        ("classifier", prod_classifier)
    ])

    prod_pipeline.fit(X, y)

    print(f"Saving final model to {MODEL_SAVE_PATH}...")
    joblib.dump(prod_pipeline, MODEL_SAVE_PATH)
    print(f"Successfully saved {MODEL_SAVE_PATH} ({os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024:.2f} MB)")

    # Self-verification across distinct emotion samples
    print("\n--- Pipeline Self-Verification Across Emotions ---")
    test_samples = {
        "happy": r"E:\ML project\Datasets\Actor_01\03-01-03-01-01-01-01.wav",
        "angry": r"E:\ML project\Datasets\Actor_01\03-01-05-01-01-01-01.wav",
        "sad": r"E:\ML project\Datasets\Actor_01\03-01-04-01-01-01-01.wav",
        "calm": r"E:\ML project\Datasets\Actor_01\03-01-02-01-01-01-01.wav",
        "surprised": r"E:\ML project\Datasets\Actor_01\03-01-08-01-01-01-01.wav",
    }
    for expected, path in test_samples.items():
        y_test_audio, sr = librosa.load(path, sr=22050, mono=True)
        feat = extract_features_from_audio(y_test_audio, sr).reshape(1, -1)
        pred = prod_pipeline.predict(feat)[0]
        prob = prod_pipeline.predict_proba(feat)[0].max() * 100
        print(f"Ground Truth: {expected:10s} -> Model Prediction: {pred:10s} (Confidence: {prob:.1f}%)")

if __name__ == "__main__":
    train_and_save()
