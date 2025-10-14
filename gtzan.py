import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------
# CONFIGURATION
# -----------------------
DATA_PATH = r"C:\Music_composer\genres"
SAMPLE_RATE = 16000
NUM_CLASSES = 10 # adjust if your dataset has 8 genres
YAMNET_MODEL_PATH = r"C:\Music_composer\yamnet"

# -----------------------
# LOAD YAMNet MODEL FROM LOCAL FOLDER
# -----------------------
print("🔹 Loading YAMNet pretrained model from local folder...")
yamnet_model = hub.KerasLayer(YAMNET_MODEL_PATH, trainable=False)
print("✅ YAMNet model loaded successfully!")

# -----------------------
# FEATURE EXTRACTION USING YAMNet
# -----------------------
def extract_yamnet_features(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        waveform = waveform.astype('float32')
        # YAMNet expects waveform shape [num_samples]
        scores, embeddings, spectrogram = yamnet_model(tf.convert_to_tensor(waveform))
        # Use mean of embeddings as feature vector
        return np.mean(embeddings.numpy(), axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# -----------------------
# LOAD DATASET
# -----------------------
def load_dataset(data_path):
    features, labels = [], []
    genres = os.listdir(data_path)
    print(f"🎶 Found genres: {genres}")

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue

        files = [f for f in os.listdir(genre_path) if f.endswith(".au") or f.endswith(".wav")]
        print(f"🎧 Extracting features for genre '{genre}' ({len(files)} files)")

        for file in files:
            file_path = os.path.join(genre_path, file)
            feature = extract_yamnet_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)

    return np.array(features), np.array(labels)

# -----------------------
# LOAD DATA
# -----------------------
print("🔹 Loading GTZAN dataset...")
X, y = load_dataset(DATA_PATH)
print(f"✅ Loaded {len(X)} audio files successfully!")

if len(X) == 0:
    raise ValueError("No audio files found! Check your DATA_PATH and file extensions.")

# -----------------------
# ENCODE LABELS
# -----------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# -----------------------
# SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.25, random_state=42, stratify=y_onehot
)
print(f"🧩 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------
# DEFINE CLASSIFIER
# -----------------------
def build_yamnet_classifier(input_shape, num_classes=NUM_CLASSES):
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
# TRAIN MODEL
# -----------------------
model = build_yamnet_classifier((X.shape[1],), NUM_CLASSES)

checkpoint = ModelCheckpoint(
    'yamnet_gtzan_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
    verbose=1
)

# -----------------------
# EVALUATE MODEL
# -----------------------
print("🔹 Evaluating model on test data...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n🎯 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

acc = accuracy_score(y_true_classes, y_pred_classes)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# -----------------------
# SAVE FINAL MODEL
# -----------------------
model.save("yamnet_gtzan_final.h5")
print("💾 Model saved as 'yamnet_gtzan_final.h5'")
