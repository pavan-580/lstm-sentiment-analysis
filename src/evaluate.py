import os

import matplotlib
matplotlib.use("Agg")  

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)


# =========================
# CONFIGURATION
# =========================

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/sentiment_lstm.keras"

X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

CLASS_NAMES = [
    "negative",
    "neutral",
    "positive"
]


# =========================
# LOAD DATA
# =========================

def load_test_data():

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    return X_test, y_test


# =========================
# LOAD MODEL
# =========================

def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


# =========================
# EVALUATE MODEL
# =========================

def evaluate_model(model, X_test, y_test):

    # Get probability predictions
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    # Convert probabilities to class numbers
    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Precision
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # F1 score
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n===== MODEL EVALUATION =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\n===== CONFUSION MATRIX =====")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    return y_pred, probabilities, cm


# =========================
# CONFUSION MATRIX PLOT
# =========================
def plot_confusion_matrix(cm):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot()

    plt.title("LSTM Sentiment Confusion Matrix")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("Confusion matrix saved successfully.")

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n===== LOADING TEST DATA =====")

    X_test, y_test = load_test_data()

    print(f"Test samples: {len(X_test)}")

    print("\n===== LOADING TRAINED MODEL =====")

    model = load_model()

    print("Model loaded successfully.")

    y_pred, probabilities, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n===== CREATING CONFUSION MATRIX =====")

    plot_confusion_matrix(cm)

    print("\n===== EVALUATION COMPLETE =====")
    print("Saved: results/confusion_matrix.png")