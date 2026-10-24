import os
import sys
import shutil
import subprocess
import traceback
import logging
import numpy as np
import librosa
import joblib
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
MODEL_PATH = os.path.join(BASE_DIR, "speech_emotion_model.pkl")

# Configure FFmpeg binary for audio decoding
try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)
    ffmpeg_standard = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_standard):
        shutil.copyfile(FFMPEG_EXE, ffmpeg_standard)
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
    logging.info(f"FFmpeg ready: {FFMPEG_EXE}")
except Exception as err:
    logging.warning(f"Could not initialize FFmpeg from imageio_ffmpeg: {err}")
    FFMPEG_EXE = "ffmpeg"

model = None

def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logging.info("speech_emotion_model.pkl loaded successfully.")
        except Exception as err:
            logging.error(f"Error loading model from {MODEL_PATH}:\n{traceback.format_exc()}")
            model = None
    return model

# Initial load
get_model()

def decode_and_load_audio(file_path, target_sr=22050):
    """
    Decodes uploaded audio (WebM/Opus, Ogg, MP3, M4A, AAC, WAV) into
    uncompressed 16-bit PCM mono WAV at 22050 Hz and loads it.
    Validates audio duration, content, and silence.
    """
    temp_decoded_wav = file_path + "_converted.wav"
    audio = None
    sr = target_sr

    try:
        # Transcode using FFmpeg
        cmd = [
            FFMPEG_EXE, "-y",
            "-i", file_path,
            "-vn",
            "-ar", str(target_sr),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            temp_decoded_wav
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(temp_decoded_wav):
            audio, sr = librosa.load(temp_decoded_wav, sr=target_sr, mono=True)
        else:
            # Fallback to direct librosa load if FFmpeg reported non-zero
            logging.warning("FFmpeg transcoding warning: %s. Falling back to librosa direct load.", res.stderr.strip())
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as decode_err:
        logging.error("Failed to decode audio file %s:\n%s", file_path, traceback.format_exc())
        raise RuntimeError(f"Could not decode audio file: {decode_err}")
    finally:
        if os.path.exists(temp_decoded_wav):
            try:
                os.remove(temp_decoded_wav)
            except Exception:
                pass

    if audio is None or len(audio) == 0:
        raise ValueError("Audio recording is empty or could not be decoded.")

    # Duration check (minimum 1.0s required by backend; frontend enforces 3s minimum)
    duration = len(audio) / sr
    if duration < 1.0:
        raise ValueError(f"Recording is too short ({duration:.1f}s). Please record at least 3 seconds of continuous speech.")

    # Silence / low audio check
    rms = float(np.sqrt(np.mean(audio ** 2)))
    max_amp = float(np.max(np.abs(audio)))
    if max_amp < 0.005 or rms < 0.0003:
        raise ValueError("The recording appears to be silent or microphone volume is too low. Please speak clearly into your microphone.")

    return audio, sr

def extract_features_from_audio(audio, sample_rate):
    """
    Extracts 236 features in exact agreement with training pipeline:
    - Active speech isolation (top_db=25 silence trimming)
    - Peak amplitude normalization
    - 40 MFCCs (mean: 40, std: 40)
    - 12 Chroma (mean: 12, std: 12)
    - 64 Mel dB (mean: 64, std: 64)
    - Zero Crossing Rate (mean: 1, std: 1)
    - Spectral Centroid (mean: 1, std: 1)
    """
    # 1. Trim leading and trailing silence to isolate active speech
    if len(audio) > 2048:
        trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if len(trimmed) >= 2048:
            audio = trimmed

    # 2. Peak normalization (normalizes dynamic range across different microphones)
    audio = librosa.util.normalize(audio)

    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.extend(np.mean(zcr, axis=1))
    features.extend(np.std(zcr, axis=1))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    features.extend(np.mean(centroid, axis=1))
    features.extend(np.std(centroid, axis=1))

    return np.array(features, dtype=np.float32).reshape(1, -1)

def extract_features(file_path):
    audio, sample_rate = decode_and_load_audio(file_path, target_sr=22050)
    return extract_features_from_audio(audio, sample_rate)

@app.route("/")
def home():
    current_model = get_model()
    return render_template("index.html", model_ready=current_model is not None)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    current_model = get_model()
    if current_model is None:
        return jsonify({
            "error": "Model file not found. Put speech_emotion_model.pkl beside app.py."
        }), 503

    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Please record or choose an audio file first."}), 400

    filename = audio_file.filename.lower()
    allowed_extensions = (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".aac", ".flac", ".opus")
    if not filename.endswith(allowed_extensions):
        return jsonify({
            "error": "Unsupported audio format. Please use WAV, MP3, OGG, WEBM, or M4A."
        }), 400

    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".webm"

    temp_path = os.path.join(BASE_DIR, f"_temp_audio_upload{ext}")
    try:
        audio_file.save(temp_path)
        features = extract_features(temp_path)
        prediction = current_model.predict(features)[0]

        probabilities = {}
        classifier = current_model.named_steps.get("classifier") if hasattr(current_model, "named_steps") else None
        classes = getattr(classifier, "classes_", None) if classifier is not None else getattr(current_model, "classes_", None)

        if classes is not None and hasattr(current_model, "predict_proba"):
            probs = current_model.predict_proba(features)[0]
            probabilities = {
                str(label): round(float(prob) * 100, 2)
                for label, prob in zip(classes, probs)
            }

        confidence = probabilities.get(str(prediction))
        if confidence is None:
            confidence = 0.0

        return jsonify({
            "emotion": str(prediction).capitalize(),
            "confidence": confidence,
            "probabilities": probabilities
        })
    except ValueError as ve:
        logging.warning("Validation error in /api/analyze: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        full_tb = traceback.format_exc()
        logging.error("Unhandled exception during /api/analyze:\n%s", full_tb)
        print(f"[ERROR /api/analyze] Traceback:\n{full_tb}", file=sys.stderr)
        return jsonify({
            "error": f"Could not analyze this audio: {str(exc)}. Try a clear recording in WAV or WebM format."
        }), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route("/api/status")
def status():
    current_model = get_model()
    return jsonify({"model_ready": current_model is not None})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
