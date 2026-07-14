"""
Text preprocessing utilities for the spam detector.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Lowercase, strip HTML/URLs/punctuation/numbers noise, collapse whitespace.
    Keeps the text as plain words, ready for vectorization.
    """
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Replace URLs with a placeholder token (their presence is itself a signal)
    text = re.sub(r"(https?://\S+|www\.\S+|\S+\.(com|net|org|io)\S*)", " urltoken ", text)

    # Replace email addresses
    text = re.sub(r"\S+@\S+", " emailtoken ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove standalone digits but keep words with mixed content (e.g. "win50")
    text = re.sub(r"\b\d+\b", " numtoken ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_features(raw_text: str) -> dict:
    """
    Engineered signals often predictive of spam, independent of the
    TF-IDF bag-of-words representation.
    """
    text = str(raw_text)
    n_chars = max(len(text), 1)

    return {
        "num_exclamations": text.count("!"),
        "num_links": len(re.findall(r"https?://|www\.", text)),
        "num_currency": len(re.findall(r"[$£€]", text)),
        "caps_ratio": sum(1 for c in text if c.isupper()) / n_chars,
        "num_digits": sum(1 for c in text if c.isdigit()),
        "length": n_chars,
    }
