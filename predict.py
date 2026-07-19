"""
Load the trained model and classify new email text.

Usage:
    python predict.py "You have WON a free prize, click here now!!!"
    python predict.py --interactive
"""

import sys
import argparse
import joblib
from scipy.sparse import hstack, csr_matrix
import pandas as pd

from preprocess import clean_text, extract_features

FEATURE_ORDER = ["num_exclamations", "num_links", "num_currency", "caps_ratio"]


def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, vectorizer, scaler


def predict_email(text: str, model, vectorizer, scaler) -> dict:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    raw_engineered = pd.DataFrame([extract_features(text)])[FEATURE_ORDER].values
    scaled_engineered = scaler.transform(raw_engineered)
    features = hstack([tfidf, csr_matrix(scaled_engineered)])

    pred = model.predict(features)[0]
    label = "SPAM" if pred == 1 else "HAM (not spam)"

    result = {"label": label}

    # Not all models support predict_proba (e.g. LinearSVC doesn't)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        result["spam_probability"] = round(float(proba[1]), 3)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Email text to classify")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    args = parser.parse_args()

    model, vectorizer, scaler = load_artifacts()

    if args.interactive:
        print("Spam Detector — type an email and press Enter (Ctrl+C to quit)\n")
        try:
            while True:
                text = input("> ")
                if not text.strip():
                    continue
                result = predict_email(text, model, vectorizer, scaler)
                print(result, "\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
        return

    if not args.text:
        print("Provide email text as an argument, or use --interactive")
        sys.exit(1)

    result = predict_email(args.text, model, vectorizer, scaler)
    print(result)


if __name__ == "__main__":
    main()
