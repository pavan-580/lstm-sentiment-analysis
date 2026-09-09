import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# CONFIGURATION
# =========================

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_lstm.keras")

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
LSTM_UNITS = 64
NUM_CLASSES = 3

EPOCHS = 20
BATCH_SIZE = 16


# =========================
# LOAD PROCESSED DATA
# =========================

def load_data():

    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)

    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test


# =========================
# BUILD LSTM MODEL
# =========================

def build_model():

    model = Sequential([
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),

        LSTM(
            LSTM_UNITS,
            return_sequences=False
        ),

        Dropout(0.3),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN MODEL
# =========================

def train_model(model, X_train, y_train):

    os.makedirs(MODEL_DIR, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    return history


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n===== LOADING PROCESSED DATA =====")

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("\n===== BUILDING LSTM MODEL =====")

    model = build_model()

    model.summary()

    print("\n===== STARTING TRAINING =====")

    history = train_model(
        model,
        X_train,
        y_train
    )

    print("\n===== TRAINING COMPLETE =====")

    print(f"Best model saved to:")
    print(MODEL_PATH)

    print("\n===== FINAL VALIDATION ACCURACY =====")

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print("\n===== TEST EVALUATION =====")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")