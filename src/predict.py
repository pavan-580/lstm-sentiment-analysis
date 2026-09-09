import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/sentiment_lstm.keras"
TOKENIZER_PATH = "data/processed/tokenizer.pkl"

MAX_LENGTH = 50

CLASS_NAMES = [
    "negative",
    "neutral",
    "positive"
]


# =========================
# LOAD MODEL AND TOKENIZER
# =========================

print("\n===== LOADING MODEL =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


print("\n===== LOADING TOKENIZER =====")

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.")


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================
# PREDICTION FUNCTION
# =========================

def predict_sentiment(review):

    # Clean text
    cleaned_review = clean_text(review)

    # Convert words into integer tokens
    sequence = tokenizer.texts_to_sequences(
        [cleaned_review]
    )

    # Pad sequence to model input length
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Get probabilities
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = np.argmax(
        probabilities
    )

    sentiment = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ] * 100

    return sentiment, confidence, probabilities


# =========================
# INTERACTIVE PREDICTION
# =========================

print("\n===== SENTIMENT PREDICTION =====")

while True:

    review = input(
        "\nEnter a customer review "
        "(type 'exit' to quit): "
    )

    if review.lower() == "exit":
        print("\nPrediction program closed.")
        break

    if not review.strip():
        print("Please enter a review.")
        continue

    sentiment, confidence, probabilities = predict_sentiment(
        review
    )

    print("\n===== RESULT =====")

    print(
        f"Review: {review}"
    )

    print(
        f"Predicted Sentiment: {sentiment.upper()}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"{class_name:10s}: "
            f"{probability * 100:.2f}%"
        )