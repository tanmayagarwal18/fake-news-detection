# Fake News Detection Using NLP and Machine Learning

A B.Tech CSE academic project (Review 1) that classifies a news
headline/article as **REAL** or **FAKE** using NLP preprocessing,
TF-IDF feature extraction, and classical machine learning models.

## 1. Description

The project takes raw news text as input, cleans and normalizes it
with an NLP pipeline, converts it into TF-IDF features, and classifies
it using one of three trained models (Logistic Regression, Multinomial
Naive Bayes, Linear SVM). The best-performing model (by F1-score) is
saved and served through a Streamlit demo app.

## 2. Problem Statement

The rapid spread of misinformation on digital platforms makes manual
fact-checking too slow to keep up with volume. This project builds an
automated, content-based (text-only) classifier that flags likely
fake news using linguistic/word-usage patterns learned from a labeled
dataset, as a first line of defense that can support (not replace)
human fact-checkers.

## 3. Features

- Reusable NLP preprocessing pipeline (`src/preprocessing.py`)
- Automated EDA with saved plots (`src/eda.py`)
- TF-IDF feature extraction (unigrams + bigrams)
- Three ML models trained & compared: Logistic Regression, Multinomial
  Naive Bayes, Linear SVM
- Full evaluation: accuracy, precision, recall, F1, confusion matrices
- Best-model auto-selection and serialization (`joblib`)
- Reusable `predict_news()` function
- Streamlit demonstration web app

## 4. Architecture

```
Raw News (CSV)
      |
Data Collection / Loading
      |
Data Cleaning (missing values, duplicates)
      |
NLP Preprocessing (lowercase, remove HTML/URLs, tokenize,
                    stopword removal, lemmatization)
      |
TF-IDF Feature Extraction (unigrams + bigrams, max_features=5000)
      |
Train/Test Split (80/20, stratified, random_state=42)
      |
Machine Learning Models (Logistic Regression / Naive Bayes / Linear SVM)
      |
Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
      |
Best Model Selection (highest F1-score)
      |
Fake / Real Prediction (src/predict.py)
      |
Streamlit Demonstration (app/app.py)
```

## 5. Folder Structure

```
fake-news-detector/
├── data/
│   ├── raw/          <- put your dataset CSV here (see below)
│   └── processed/    <- reserved for any cached processed data
├── notebooks/        <- optional Jupyter exploration
├── src/
│   ├── preprocessing.py   NLP cleaning pipeline
│   ├── eda.py              exploratory data analysis + plots
│   ├── train.py             TF-IDF + train/evaluate/save 3 models
│   ├── predict.py            load saved model, predict_news()
│   └── utils.py               shared paths/logging helpers
├── models/            <- best_model.joblib, tfidf_vectorizer.joblib
├── app/
│   └── app.py         <- Streamlit demo UI
├── results/            <- plots, metrics, comparison tables (generated)
├── docs/
│   ├── literature_survey.md
│   └── review1_ppt_content.md
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Dataset

**Recommended dataset:** *Fake and Real News Dataset* (Kaggle, uploaded
by Clément Bisaillon), which packages the **ISOT Fake News Dataset**
originally released by the University of Victoria's ISOT Research Lab.

- **Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Original research:** Ahmed, Traore & Saad (2017, 2018) -- see
  `docs/literature_survey.md`, Paper 1.
- **Format:** two CSVs, `Fake.csv` and `True.csv`, each with columns
  `title, text, subject, date`.
- **Why this dataset:** it is free, widely used in fake-news-detection
  coursework and papers (making your results comparable to published
  baselines), reasonably large (tens of thousands of articles per
  class), and requires no special access requests.
- **Limitations to disclose in your report:** the "real" articles are
  sourced almost entirely from Reuters and the "fake" articles from a
  specific set of flagged sites, so a trained model may partly learn
  *source style* rather than pure fake-vs-real reasoning; it may not
  generalize well to news outside those sources/time period (2016-2017
  U.S.-centric political news). Mention this explicitly as future work
  ("domain generalization") during your viva.

**We do not fabricate dataset statistics.** Exact row counts, class
balance, and vocabulary size for your copy of the dataset will be
written to `results/eda_summary.json` after you run `src/eda.py`.

### Setting up the dataset

1. Download `Fake.csv` and `True.csv` from the Kaggle link above.
2. Place both files in `data/raw/`.
3. Combine them into a single labeled file:

```python
import pandas as pd
fake = pd.read_csv("data/raw/Fake.csv"); fake["label"] = "FAKE"
real = pd.read_csv("data/raw/True.csv"); real["label"] = "REAL"
pd.concat([fake, real], ignore_index=True).to_csv("data/raw/news.csv", index=False)
```

This produces `data/raw/news.csv` with columns `title, text, subject,
date, label`, which is exactly what the default script arguments
expect.

**Using a different dataset?** Every script accepts `--text-col`,
`--title-col`, and `--label-col` arguments so you can point them at
whatever column names your CSV actually has, e.g.:

```
python src/train.py --file mydata.csv --text-col article --label-col target
```

## 7. Technologies

Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, NLTK,
joblib, Streamlit.

## 8. Installation

```bash
git clone <your-repo-url>
cd fake-news-detector
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run of `src/preprocessing.py` (imported by every other
script) will automatically download the small NLTK resources it needs
(punkt, stopwords, wordnet) — this requires internet access once.

## 9. How to Run

Step-by-step (also see `docs/review1_ppt_content.md` for what
screenshot to take at each stage):

```bash
# 1. Place your dataset (see section 6) at data/raw/news.csv

# 2. Exploratory Data Analysis -> figures + eda_summary.json in results/
python src/eda.py --file news.csv

# 3. Train all 3 models, evaluate, save the best one
python src/train.py --file news.csv

# 4. Command-line prediction check
python src/predict.py "Scientists confirm shocking new coffee cure for everything"

# 5. Launch the demo web app
streamlit run app/app.py
```

## 10. Model Training Details

- Split: 80% train / 20% test, stratified by label, `random_state=42`.
- Features: TF-IDF over title+text, unigrams+bigrams, `max_features=5000`.
- Models compared: Logistic Regression, Multinomial Naive Bayes, Linear SVM.
- Model selection metric: **F1-score** (balances precision and recall;
  see `src/train.py` docstring for why this matters more than raw
  accuracy for fake-news detection specifically -- false negatives
  (missed fake news) and false positives (wrongly flagging real news)
  are both costly).

## 11. Prediction

```python
from src.predict import predict_news
result = predict_news("Some headline or article text")
print(result)
# {'label': 'FAKE', 'confidence': 94.32, 'is_calibrated': True, 'model_used': 'Logistic Regression'}
```

`is_calibrated` is `False` when the chosen best model is the Linear
SVM, because `LinearSVC` has no native `predict_proba`; in that case
`confidence` is a decision-score-based approximation, not a true
probability, and the UI/CLI both label it accordingly.

## 12. Streamlit Application

`app/app.py` loads the already-trained model + vectorizer from
`models/` (via `src/predict.py`) — it does **not** retrain on each
request. Run with `streamlit run app/app.py`.

## 13. Results

Populated automatically in `results/` after running `src/train.py`:
`model_comparison.csv`, `model_comparison.json`, one confusion-matrix
PNG per model, and one `classification_report_*.txt` per model. No
numbers in this README are invented — check those files for your
actual run's results.

## 14. Future Improvements

- Word2Vec/GloVe embeddings + LSTM, or a fine-tuned DistilBERT model,
  as a stronger but heavier alternative to TF-IDF + classical ML
  (see Section 13 "Optional Advanced Feature" of the project brief —
  planned as post-Review-1 work).
- Cross-domain evaluation on a second, independent dataset to test
  generalization beyond the training source's writing style.
- Explainability (e.g. SHAP/LIME on TF-IDF features) to show *why* an
  article was flagged, not just the label.
- Multilingual support.

## 15. Team

- Name: _____________________
- Roll No.: _____________________
- Guide: _____________________
- Department: CSE
