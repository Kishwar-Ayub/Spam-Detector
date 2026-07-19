"""
Train a spam email classifier.

Usage:
    python train.py                     # uses data/emails.csv
    python train.py --data path/to.csv  # use your own dataset (columns: label,text)
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocess import clean_text, extract_features

FEATURE_ORDER = ["num_exclamations", "num_links", "num_currency", "caps_ratio"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must have 'label' and 'text' columns.")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)
    return df


def engineered_matrix(texts) -> np.ndarray:
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows)[FEATURE_ORDER].values


def build_feature_matrix(texts, vectorizer, scaler, fit=False):
    """
    Combines TF-IDF text features with engineered signals (exclamation
    count, link count, etc). The engineered features are scaled to roughly
    the same 0-1 range as TF-IDF values before combining — without this,
    raw counts like message length (which can be in the hundreds) swamp
    the TF-IDF signal and the model ends up reacting mostly to length
    rather than actual content.
    """
    cleaned = [clean_text(t) for t in texts]

    if fit:
        tfidf = vectorizer.fit_transform(cleaned)
    else:
        tfidf = vectorizer.transform(cleaned)

    raw_engineered = engineered_matrix(texts)
    if fit:
        scaled_engineered = scaler.fit_transform(raw_engineered)
    else:
        scaled_engineered = scaler.transform(raw_engineered)

    combined = hstack([tfidf, csr_matrix(scaled_engineered)])
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/emails.csv")
    parser.add_argument("--model", default="logistic_regression",
                         choices=["naive_bayes", "logistic_regression", "svm"])
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows. Class balance:\n{df['label'].value_counts()}\n")

    y = (df["label"] == "spam").astype(int).values
    X_text = df["text"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    scaler = MinMaxScaler()

    print("Building features (TF-IDF + scaled engineered signals)...")
    X_train = build_feature_matrix(X_train_text, vectorizer, scaler, fit=True)
    X_test = build_feature_matrix(X_test_text, vectorizer, scaler, fit=False)

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
        "svm": LinearSVC(class_weight="balanced"),
    }
    model = models[args.model]

    print(f"Training model: {args.model} ...")
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"\n5-fold CV F1 scores: {np.round(cv_scores, 3)}")
    print(f"Mean CV F1: {cv_scores.mean():.3f}")

    preds = model.predict(X_test)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"F1 Score: {f1_score(y_test, preds):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    print("Confusion Matrix (rows=true, cols=predicted) [ham, spam]:")
    print(confusion_matrix(y_test, preds))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/spam_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("\nSaved model/spam_model.pkl, model/vectorizer.pkl, and model/scaler.pkl")


if __name__ == "__main__":
    main()
