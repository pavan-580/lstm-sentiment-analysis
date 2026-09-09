import os
import re
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st

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
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Customer Sentiment AI",
    page_icon="🧠",
    layout="centered"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================
# LOAD TOKENIZER
# =========================

@st.cache_resource
def load_tokenizer():

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


model = load_model()
tokenizer = load_tokenizer()


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
# SENTIMENT PREDICTION
# =========================

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_review]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

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
# USER INTERFACE
# =========================

st.title("🧠 Customer Sentiment AI")

st.write(
    "Analyze the sentiment of a customer review "
    "using an LSTM deep learning model."
)

st.divider()

review = st.text_area(
    "Enter a customer review:",
    placeholder="Example: The product is excellent and I am very happy with it.",
    height=151
)


if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning(
            "Please enter a customer review."
        )

    else:

        sentiment, confidence, probabilities = (
            predict_sentiment(review)
        )

        st.subheader("Prediction")

        if sentiment == "positive":

            st.success(
                f"😊 POSITIVE"
            )

        elif sentiment == "negative":

            st.error(
                f"😞 NEGATIVE"
            )

        else:

            st.info(
                f"😐 NEUTRAL"
            )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )

        st.divider()

        st.subheader(
            "Class Probabilities"
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            st.write(
                f"**{class_name.capitalize()}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


# =========================
# PROJECT INFORMATION
# =========================

st.divider()

st.caption(
    "Model: LSTM | "
    "Task: 3-class sentiment classification | "
    "Classes: Negative, Neutral, Positive"
)