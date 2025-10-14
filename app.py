import os
import io
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Imports for MusicGen
from transformers import pipeline
import scipy.io.wavfile as wavfile
import base64

# Imports for Genre Classification
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import tempfile 

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app) 

# --- CLASSIFICATION CONSTANTS (MUST MATCH gtzan_genre_classifier.py) ---
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

# --- MODEL LOADING (Only runs once when the server starts) ---

# 1. Load MusicGen
try:
    print("Attempting to load MusicGen model (facebook/musicgen-small)...")
    music_generator = pipeline("text-to-audio", model="facebook/musicgen-small", device=-1) 
    print("MusicGen 'small' model loaded successfully.")
except Exception as e:
    print("-" * 50)
    print("FATAL MODEL LOADING FAILURE: MusicGen model failed to load.")
    print(f"Error details: {e}")
    print("-" * 50)

# 2. Load GTZAN Classifier
try:
    print(f"Attempting to load GTZAN Classifier model from {CLASSIFIER_MODEL_PATH}...")
    classifier_model = load_model(CLASSIFIER_MODEL_PATH)
    print("GTZAN Classifier Model loaded successfully.")
except Exception as e:
    print("-" * 50)
    print(f"ERROR: Could not load classifier model at '{CLASSIFIER_MODEL_PATH}'. Classification endpoint will be disabled.")
    print(f"Details: {e}")
    print("-" * 50)

# --- CORE GENRE PREDICTION FUNCTION ---

def predict_genre_from_file(file_path):
    """
    Processes an audio file, extracts features, and returns the predicted genre string.
    """
    if classifier_model is None:
        return None, "Model not initialized."

    try:
        # 1. Load Audio and ensure it's at least 30 seconds
        signal, sr = librosa.load(file_path, sr=CLASSIFIER_SAMPLE_RATE)
        
        # Trim or pad signal to 30 seconds
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        elif len(signal) < SAMPLES_PER_TRACK:
            padding = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')

        segments = []
        
        # 2. Extract Features for Each Segment
        for s in range(NUM_SEGMENTS):
            start_sample = SAMPLES_PER_SEGMENT * s
            end_sample = start_sample + SAMPLES_PER_SEGMENT
            
            mfcc = librosa.feature.mfcc(
                y=signal[start_sample:end_sample],
                sr=sr,
                n_mfcc=NUM_MFCC,
                hop_length=HOP_LENGTH
            ).T
            
            if mfcc.shape[0] == NUM_MFCC_VECTORS_PER_SEGMENT:
                segments.append(mfcc[..., np.newaxis])
        
        if not segments:
            return None, "All audio segments failed feature extraction."

        X_segments = np.array(segments)

        # 3. Predict and Aggregate
        predictions = classifier_model.predict(X_segments, verbose=0)
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        
        predicted_genre = GENRE_MAPPING[predicted_index]
        confidence = avg_prediction[predicted_index] * 100
        
        return predicted_genre, f"Confidence: {confidence:.2f}%"

    except Exception as e:
        return None, f"Prediction error during feature extraction or model call: {e}"

# --- ROUTES ---

@app.route('/')
def serve_index():
    """Serves the index.html file from the templates folder."""
    return render_template('index.html')

@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    """Endpoint to receive an audio file, classify its genre, and return the result."""
    if classifier_model is None:
        return jsonify({"status": "error", "message": "Genre Classifier model is not loaded."}), 503
    
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided in request."}), 400

    audio_file = request.files['audio_file']
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_{audio_file.filename}")
    os.close(temp_fd)
    
    try:
        audio_file.save(temp_path)
        
        # Get prediction
        genre, message = predict_genre_from_file(temp_path)
        
        if genre:
            return jsonify({
                "status": "success",
                "genre": genre,
                "detail": message,
                "mapping": GENRE_MAPPING
            })
        else:
            return jsonify({"status": "error", "message": f"Classification failed. {message}"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server processing error: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/generate-music', methods=['POST'])
def generate_music_api():
    """Receives prompt/duration, generates music, and returns it as Base64 encoded audio."""
    if not music_generator:
        return jsonify({"error": "AI model is not ready or failed to load on the server."}), 503

    try:
        data = request.json
        base_prompt = data.get('prompt')
        duration = int(data.get('duration', 5))
        inferred_genre = data.get('inferred_genre') # NEW: Get genre from frontend
        
        # Input validation
        if not base_prompt:
            return jsonify({"error": "No music prompt provided."}), 400
        if duration < 3 or duration > 15:
            return jsonify({"error": "Duration must be between 3 and 15 seconds."}), 400 

        # NEW: Enrich the prompt if a genre was inferred
        if inferred_genre:
            # Format: "Genre music, [original prompt]"
            enriched_prompt = f"{inferred_genre} music, {base_prompt}"
            print(f"MusicGen prompt enriched: {enriched_prompt}")
        else:
            enriched_prompt = base_prompt
            print(f"MusicGen prompt used: {base_prompt}")

        max_new_tokens = int(duration * 30)
        
        # 1. Generate Music using MusicGen
        music = music_generator(
            enriched_prompt, # Use the enriched prompt
            generate_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": True
            }
        )
        
        audio_data = music["audio"]
        sampling_rate = music["sampling_rate"]
        
        # 2. Convert to WAV in memory buffer
        buffer = io.BytesIO()
        audio_int16 = (audio_data * 32767).astype(np.int16) 
        wavfile.write(buffer, rate=sampling_rate, data=audio_int16.squeeze())
        buffer.seek(0)
        
        # 3. Encode WAV data to Base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print("Music generated and encoded successfully.")
        
        # 4. Return the Base64 string
        return jsonify({
            "status": "success",
            "audio_base64": audio_base64,
            "mime_type": "audio/wav",
            "final_prompt": enriched_prompt # Return the final prompt used
        })

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": f"An error occurred during music generation: {e}"}), 500

# --- Run Server ---
if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)