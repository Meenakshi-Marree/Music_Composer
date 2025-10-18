import os
import io
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import scipy.io.wavfile as wavfile
import base64
from tensorflow.keras.models import load_model
import numpy as np
import librosa

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app) 

# --- CLASSIFICATION CONSTANTS ---
CLASSIFIER_MODEL_PATH = "yamnet_gtzan_final.h5"
CLASSIFIER_SAMPLE_RATE = 22050
NUM_MFCC = 20
HOP_LENGTH = 512
NUM_SEGMENTS = 10 
TRACK_DURATION = 30
SAMPLES_PER_TRACK = CLASSIFIER_SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
NUM_MFCC_VECTORS_PER_SEGMENT = int(np.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH))
GENRE_MAPPING = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- GLOBAL MODEL INSTANCES ---
music_generator = None
classifier_model = None

# --- LOAD MODELS ---
try:
    print("Loading MusicGen model...")
    music_generator = pipeline("text-to-audio", model="facebook/musicgen-small", device=-1)
    print("MusicGen loaded successfully.")
except Exception as e:
    print(f"Error loading MusicGen: {e}")

try:
    print(f"Loading GTZAN classifier from {CLASSIFIER_MODEL_PATH}...")
    classifier_model = load_model(CLASSIFIER_MODEL_PATH)
    print("Classifier loaded successfully.")
except Exception as e:
    print(f"Error loading classifier: {e}")

# --- CORE FUNCTIONS ---
def predict_genre_from_file(file_path):
    if classifier_model is None:
        return None, "Classifier not initialized."
    try:
        signal, sr = librosa.load(file_path, sr=CLASSIFIER_SAMPLE_RATE)
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            signal = np.pad(signal, (0, max(0, SAMPLES_PER_TRACK - len(signal))), 'constant')

        segments = []
        for s in range(NUM_SEGMENTS):
            start_sample = SAMPLES_PER_SEGMENT * s
            end_sample = start_sample + SAMPLES_PER_SEGMENT
            mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sr, n_mfcc=NUM_MFCC, hop_length=HOP_LENGTH).T
            if mfcc.shape[0] == NUM_MFCC_VECTORS_PER_SEGMENT:
                segments.append(mfcc[..., np.newaxis])

        if not segments:
            return None, "All segments failed."
        X_segments = np.array(segments)
        predictions = classifier_model.predict(X_segments, verbose=0)
        avg_pred = np.mean(predictions, axis=0)
        idx = np.argmax(avg_pred)
        genre = GENRE_MAPPING[idx]
        confidence = avg_pred[idx] * 100
        return genre, f"Confidence: {confidence:.2f}%"
    except Exception as e:
        return None, f"Error: {e}"

# --- ROUTES ---

@app.route('/')
def serve_index():
    return render_template('home.html')

@app.route('/compose')
def serve_home():
    return render_template('index.html')

@app.route('/library')
def serve_library():
    return render_template('library.html')

@app.route('/playlist')
def playlist():
    return render_template('playlist.html')

@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    if classifier_model is None:
        return jsonify({"status": "error", "message": "Classifier not loaded."}), 503
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided."}), 400

    audio_file = request.files['audio_file']
    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_{audio_file.filename}")
    os.close(temp_fd)
    try:
        audio_file.save(temp_path)
        genre, msg = predict_genre_from_file(temp_path)
        if genre:
            return jsonify({"status":"success","genre":genre,"detail":msg,"mapping":GENRE_MAPPING})
        return jsonify({"status":"error","message":msg}),500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/generate-music', methods=['POST'])
def generate_music_api():
    if not music_generator:
        return jsonify({"error": "MusicGen not loaded."}), 503

    try:
        data = request.json
        base_prompt = data.get('prompt')
        duration = int(data.get('duration', 5))
        inferred_genre = data.get('inferred_genre')

        if not base_prompt:
            return jsonify({"error": "No prompt provided."}), 400
        if duration < 3 or duration > 15:
            return jsonify({"error": "Duration must be 3-15s."}), 400

        # Enrich prompt with inferred genre
        enriched_prompt = f"{inferred_genre} music, {base_prompt}" if inferred_genre else base_prompt
        print(f"[DEBUG] Prompt used for generation: {enriched_prompt}")
        print(f"[DEBUG] Duration: {duration}s")

        # Use safer max_new_tokens for CPU
        max_new_tokens = int(duration * 30)  # ~30 tokens per second
        print(f"[DEBUG] max_new_tokens: {max_new_tokens}")

        # Generate music
        music = music_generator(
            enriched_prompt,
            generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": True}
        )

        audio_data = music.get("audio")
        sampling_rate = music.get("sampling_rate", 32000)  # default 32k

        if audio_data is None or audio_data.size == 0:
            return jsonify({"error": "Generated audio is empty. Check prompt or model."}), 500

        print(f"[DEBUG] Generated audio shape: {audio_data.shape}, duration: {len(audio_data)/sampling_rate:.2f} sec")

        # Convert to WAV in memory
        buffer = io.BytesIO()
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(buffer, rate=sampling_rate, data=audio_int16.squeeze())
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "audio_base64": audio_base64,
            "mime_type": "audio/wav",
            "final_prompt": enriched_prompt
        })

    except Exception as e:
        print(f"[ERROR] During generation: {e}")
        return jsonify({"error": str(e)}), 500


# --- RUN SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
