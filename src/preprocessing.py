import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data/raw/customer_reviews_improved.csv"

PROCESSED_DIR = "data/processed"
TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "tokenizer.pkl")

MAX_WORDS = 5000
MAX_LENGTH = 50
TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================
# LOAD DATASET
# =========================

def load_dataset():
    """Load the raw customer review dataset."""
    return pd.read_csv(DATA_PATH)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    """Clean a review for NLP processing."""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# LABEL ENCODING
# =========================

def encode_labels(labels):
    """
    Convert sentiment labels to numbers.

    negative -> 0
    neutral  -> 1
    positive -> 2
    """

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    return labels.map(label_map)


# =========================
# PREPROCESS DATA
# =========================

def preprocess_data(df):

    # Clean text
    df["clean_review"] = df["review"].apply(clean_text)

    # Encode labels
    df["label"] = encode_labels(df["sentiment"])

    # Remove rows that could not be encoded
    df = df.dropna(subset=["label"])

    X = df["clean_review"]
    y = df["label"].astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Create tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    # IMPORTANT:
    # Fit tokenizer only on training data
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    # Pad sequences
    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return (
        X_train_padded,
        X_test_padded,
        y_train,
        y_test,
        tokenizer
    )


# =========================
# SAVE PROCESSED DATA
# =========================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    tokenizer
):

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save arrays as NumPy files
    X_train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
    X_test_path = os.path.join(PROCESSED_DIR, "X_test.npy")
    y_train_path = os.path.join(PROCESSED_DIR, "y_train.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.npy")

    import numpy as np

    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    # Save tokenizer
    with open(TOKENIZER_PATH, "wb") as file:
        pickle.dump(tokenizer, file)

    print("\n===== PROCESSED DATA SAVED =====")
    print(f"X_train: {X_train_path}")
    print(f"X_test : {X_test_path}")
    print(f"y_train: {y_train_path}")
    print(f"y_test : {y_test_path}")
    print(f"Tokenizer: {TOKENIZER_PATH}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n===== LOADING DATASET =====")

    df = load_dataset()

    print(f"Original rows: {len(df)}")

    print("\n===== PREPROCESSING =====")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    ) = preprocess_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    print("\n===== VOCABULARY =====")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")

    print("\n===== SAMPLE TOKENIZED REVIEW =====")
    print(X_train[0])

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    )

    print("\n===== PREPROCESSING COMPLETE =====")

