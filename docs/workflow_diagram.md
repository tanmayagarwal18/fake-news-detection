# Workflow Diagrams

A rendered version of both diagrams is also in `Review1_Presentation.pptx`
(Slide 12) and as a standalone image at `results/workflow_diagram.png`.

## 1. Main System Architecture

```
 Raw News
    │
    ▼
 Data Collection  (Kaggle Fake/Real News CSV)
    │
    ▼
 Data Cleaning  (missing values, duplicate removal)
    │
    ▼
 NLP Preprocessing  (lowercase → strip HTML/URLs → strip punctuation
                      → tokenize → remove stopwords → lemmatize)
    │
    ▼
 TF-IDF Feature Extraction  (unigrams + bigrams, max_features=5000)
    │
    ▼
 Train/Test Split  (80/20, stratified, random_state=42)
    │
    ▼
 Machine Learning Models  (Logistic Regression | Naive Bayes | Linear SVM)
    │
    ▼
 Model Evaluation  (Accuracy, Precision, Recall, F1, Confusion Matrix)
    │
    ▼
 Best Model Selection  (highest F1-score)
    │
    ▼
 Fake / Real Prediction  (src/predict.py -> predict_news())
    │
    ▼
 Streamlit Demonstration  (app/app.py)
```

## 2. Detailed NLP Preprocessing Sub-Workflow

```
 Raw text (title + article)
    │
    ▼
 Lowercasing
    │
    ▼
 Remove HTML tags
    │
    ▼
 Remove URLs
    │
    ▼
 Remove punctuation & non-alphabetic characters
    │
    ▼
 Tokenization  (NLTK word_tokenize)
    │
    ▼
 Stopword removal  (NLTK English stopword list)
    │
    ▼
 Lemmatization  (NLTK WordNetLemmatizer)
    │
    ▼
 Cleaned token string  -> ready for TfidfVectorizer
```

This exactly matches the implementation in `src/preprocessing.py`
(`clean_text()`), so the diagram stays accurate as long as that file
doesn't change.
