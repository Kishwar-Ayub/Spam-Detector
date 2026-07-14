# AI-Powered Spam Email Detector

A complete, runnable starter project: TF-IDF + engineered features + a classic
ML classifier (Naive Bayes / Logistic Regression / Linear SVM) to detect spam.

## Project structure
```
spam-detector/
├── data/
│   ├── generate_sample_data.py   # creates a small demo dataset
│   └── emails.csv                # generated demo dataset (label, text)
├── preprocess.py                 # text cleaning + engineered features
├── train.py                      # train, evaluate, save the model
├── predict.py                    # classify new email text
├── model/                        # saved model + vectorizer (after training)
├── requirements.txt
└── README.md
```

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train
```bash
python train.py
```
This trains on `data/emails.csv` and prints accuracy, F1, a classification
report, and a confusion matrix. It saves `model/spam_model.pkl` and
`model/vectorizer.pkl`.

Try a different model:
```bash
python train.py --model logistic_regression
python train.py --model svm
```

## 3. Predict on new text
```bash
python predict.py "Congratulations! You've won a free prize, click here now!!!"
```
Or interactively:
```bash
python predict.py --interactive
```

## ⚠️ Important: the included dataset is a DEMO, not real data
`data/emails.csv` is **synthetically generated from templates** (see
`generate_sample_data.py`) so you can run the whole pipeline immediately.
Because the templates are repetitive, the model scores near-perfect accuracy
on it — that is a property of the toy data, not a signal that the model is
production-ready. Real emails are much messier.

**Before treating this as a real project, swap in real data:**

1. **SMS Spam Collection** (Kaggle, small and clean — easiest first upgrade)
2. **Enron-Spam dataset** (real emails, more realistic distribution)
3. **SpamAssassin public corpus** (raw email format, good for practicing
   header/attachment parsing too)

To use your own dataset, just point `train.py` at a CSV with `label` and
`text` columns:
```bash
python train.py --data path/to/your_dataset.csv
```

## How it works
1. **Cleaning** (`preprocess.clean_text`): lowercases text, strips HTML,
   replaces URLs/emails with placeholder tokens (their *presence* is a
   signal even after removal), strips punctuation and standalone numbers.
2. **Engineered features** (`preprocess.extract_features`): exclamation
   count, link count, currency symbol count, ALL-CAPS ratio, digit count,
   message length — classic spam "tells" that complement bag-of-words.
3. **TF-IDF vectorization**: converts cleaned text into weighted word/bigram
   frequency vectors (top 5,000 features).
4. **Classifier**: TF-IDF + engineered features are combined and fed into
   your chosen model. Naive Bayes is a strong, fast baseline for text;
   Logistic Regression and Linear SVM often edge it out and give you class
   weighting for imbalanced data.
5. **Evaluation**: 5-fold cross-validation F1 plus a held-out test set with
   full precision/recall/F1 and a confusion matrix — pay special attention to
   false positives (ham marked as spam), since those are the costly mistakes
   in a real inbox.

## Next steps to level up
- Swap in a real dataset (see above) and re-tune `min_df`/`max_features`
- Try `GridSearchCV` to tune hyperparameters
- Handle class imbalance with `class_weight="balanced"` (already wired up
  for logistic regression / SVM) or SMOTE oversampling
- Try a fine-tuned transformer (e.g. DistilBERT) for a deep-learning upgrade
- Add explainability: inspect `model.coef_` (logistic regression) or
  `feature_log_prob_` (Naive Bayes) against `vectorizer.get_feature_names_out()`
  to see which words most drive spam predictions
- Wrap `predict.py`'s logic in a small Flask/FastAPI endpoint to serve
  predictions over HTTP, or connect it to the Gmail API to classify real
  incoming mail
