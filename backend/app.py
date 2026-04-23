import os
import pickle
import traceback

from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "cnn_model.h5")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer.pkl")

MAX_LEN = 100  # change this only if your training used a different max length

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


def is_realtime_claim(text):
    text = text.lower()
    keywords = [
        "today", "just now", "live", "currently", "won today",
        "breaking", "latest", "now", "happening now", "this match"
    ]
    return any(word in text for word in keywords)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("news_text", "").strip()

        if user_text:
            if is_realtime_claim(user_text):
                result = "Cannot verify live or real-time news"
                confidence = None
            else:
                try:
                    processed = preprocess_text(user_text)
                    pred = model.predict(processed)[0][0]

                    # adjust this logic if your label mapping is opposite
                    if pred >= 0.5:
                        result = "Likely Real"
                        confidence = round(float(pred) * 100, 2)
                    else:
                        result = "Likely Fake"
                        confidence = round((1 - float(pred)) * 100, 2)

                except Exception:
                    result = "Error while making prediction"
                    print(traceback.format_exc())

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        user_text=user_text
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)