from flask import Flask, render_template, request, redirect
import tensorflow as tf
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "fake-news-secret-key"

# ----------------------------
# Load model & tokenizer
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "cnn_model.h5")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=100, padding="post", truncating="post"
    )
    return padded

# ----------------------------
# Optional: detect real-time claims
# ----------------------------
def is_realtime_claim(text):
    text = text.lower()
    keywords = ["today", "live", "just now", "breaking", "currently"]
    return any(word in text for word in keywords)

# ----------------------------
# MAIN ROUTE (same page prediction)
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("news", "").strip()

        if user_text:
            if is_realtime_claim(user_text):
                result = "Cannot verify live or real-time news"
                confidence = None
            else:
                try:
                    processed = preprocess_text(user_text)
                    pred = model.predict(processed)[0][0]

                    if pred >= 0.5:
                        result = "Likely Real"
                        confidence = round(float(pred) * 100, 2)
                    else:
                        result = "Likely Fake"
                        confidence = round((1 - float(pred)) * 100, 2)

                except Exception as e:
                    print(e)
                    result = "Error while making prediction"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        user_text=user_text
    )

# ----------------------------
# OPTIONAL ROUTES (to avoid 404 errors)
# ----------------------------
@app.route("/history")
def history():
    return "History page (you can implement later)"

@app.route("/logout")
def logout():
    return redirect("/")

# ----------------------------
# Run (for local testing)
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)