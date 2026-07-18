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
├── email_utils.py                # parses .eml files into headers + body
├── url_utils.py                  # extracts URLs + Google Safe Browsing check
├── threat_intel.py               # VirusTotal, urlscan.io, MXToolbox checks
├── train.py                      # train, evaluate, save the model
├── predict.py                    # classify new email text (CLI)
├── app.py                        # Streamlit web UI
├── .streamlit/
│   └── secrets.toml.example      # template for your API key (copy, don't commit real key)
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

## 3. Predict on new text (command line)
```bash
python predict.py "Congratulations! You've won a free prize, click here now!!!"
```
Or interactively:
```bash
python predict.py --interactive
```

## 4. Real-time link check (Google Safe Browsing)
On top of the ML text classifier, the app can cross-check any links found in
an email against **Google Safe Browsing** — the same live, constantly-updated
threat database Chrome uses to block phishing/malware sites. This gives you
a genuine real-time, third-party signal separate from your own model.

**Get a free API key:**
1. Go to console.cloud.google.com and create a project (or use an existing one)
2. In the search bar, find and enable the **"Safe Browsing API"**
3. Go to **APIs & Services > Credentials** → **Create Credentials** → **API key**
4. Copy the generated key

**Add the key locally (never commit it to GitHub):**
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Replace the placeholder with your real key:
   ```toml
   SAFE_BROWSING_API_KEY = "your-actual-key-here"
   ```
3. Add `.streamlit/secrets.toml` to your `.gitignore` — only the `.example`
   file is meant to be committed

**Add the key on Streamlit Community Cloud:**
1. Go to **share.streamlit.io**, find your app, click **⋮ → Settings**
2. Open the **"Secrets"** tab
3. Paste:
   ```toml
   SAFE_BROWSING_API_KEY = "your-actual-key-here"
   ```
4. Save — the app restarts with the key available via `st.secrets`

Without a key configured, the app still works — it just lists the extracted
links without a live safety verdict, and tells you how to enable the check.

**Note:** Safe Browsing's free Lookup API is for non-commercial use. If this
ever becomes a commercial product, Google's paid **Web Risk API** is the
equivalent for commercial use.

## 5. More threat-intel sources (VirusTotal, urlscan.io, MXToolbox)
On top of Google Safe Browsing, the app can cross-check links and sender
domains against three more established platforms. All are optional — each
just silently stays off until you add its key.

**VirusTotal** — checks a URL's reputation across 70+ antivirus/security
engines.
1. Sign up free at virustotal.com, then find your API key in your account's
   API key page
2. Add as a secret: `VIRUSTOTAL_API_KEY = "your-key-here"`
3. Free tier: ~4 requests/minute, non-commercial use only. If you're
   checking messages with several links at once, you may hit this limit —
   the app will show an "error" badge for that source if so, the others
   keep working independently.

**urlscan.io** — looks up whether a domain has prior scan history and any
past malicious verdicts. (To stay fast, this checks *existing* scans only —
submitting a brand-new scan takes 30+ seconds, too slow for an inline check.)
1. Sign up free at urlscan.io, create an API key from your account page
2. Add as a secret: `URLSCAN_API_KEY = "your-key-here"`

**MXToolbox** — checks whether the *sender's domain* (from the `.eml` file's
From header) is on any email blacklist. Shown only on the "Upload case file"
tab, since pasted text has no sender header.
1. Create a free account at mxtoolbox.com and request an API key from their
   API reference page
2. Add as a secret: `MXTOOLBOX_API_KEY = "your-key-here"`
3. Note: MXToolbox's docs have been inconsistent about whether free accounts
   still get API access or if it now requires a paid plan — check current
   terms on their site. If your key gets rejected, the app shows a clear
   error rather than crashing, and the other three checks keep working.

**Adding secrets on Streamlit Community Cloud:** share.streamlit.io → your
app → **⋮ → Settings → Secrets**, then paste all the keys you're using
together, e.g.:
```toml
SAFE_BROWSING_API_KEY = "..."
VIRUSTOTAL_API_KEY = "..."
URLSCAN_API_KEY = "..."
MXTOOLBOX_API_KEY = "..."
```
You don't need all four — add only the ones you've signed up for.

## 6. Web UI
```bash
streamlit run app.py
```
This opens a browser tab with a text box — paste in an email and click
"Check for spam" to see the prediction and spam probability. Requires
`model/spam_model.pkl` and `model/vectorizer.pkl` to already exist (run
`train.py` first).

The UI has two tabs:
- **Paste text** — type or paste subject/body directly.
- **Upload email file (.eml)** — upload a raw email file (most mail clients
  let you export/"Save As" the raw `.eml` message). This extracts and
  displays: **From, To, Subject, Date, Message-ID, Received, and
  Authentication-Results** (SPF/DKIM/DMARC), then runs the subject + body
  text through the same spam classifier. A failed Authentication-Results
  check is flagged separately as an independent red flag, since header
  authentication and text-based spam detection catch different things.

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
