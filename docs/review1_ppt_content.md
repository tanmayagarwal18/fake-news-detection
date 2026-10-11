# Review 1 Presentation Content (16 Slides)

Concise, presentation-ready content. A matching `.pptx` file is also
provided (`Review1_Presentation.pptx`) with this content already laid
out — use this document mainly for the speaker notes.

---

### Slide 1 — Title
- **Fake News Detection Using NLP and Machine Learning**
- Subtitle: B.Tech CSE Project — Review 1
- Your name, roll number, guide name, department
- *Speaker notes:* Introduce yourself and the one-line goal: "automatically classify news text as real or fake using NLP and machine learning."

### Slide 2 — Introduction
- Misinformation spreads faster than fact-checking can keep up
- NLP + ML can flag suspicious content automatically, at scale
- This project builds a text-based REAL/FAKE classifier
- *Diagram:* a simple icon row: news source → social media → reader, with a "?" over shared content
- *Speaker notes:* Set context — why this matters now (social media scale, speed of spread).

### Slide 3 — Problem Statement
- Manual fact-checking cannot scale to the volume of daily content
- Need: an automated, content-based first-pass classifier
- Goal: given a headline/article, predict REAL or FAKE with a confidence score
- *Speaker notes:* Be precise that this is a content-based classifier, not a full fact-checking system.

### Slide 4 — Objectives
- Build an NLP preprocessing pipeline for news text
- Extract TF-IDF features from cleaned text
- Train and compare 3 ML classifiers
- Deploy the best model behind a simple demo UI
- *Speaker notes:* These four objectives map directly to the four architecture stages on Slide 12.

### Slide 5 — Motivation / Existing System
- Manual moderation/fact-checking: accurate but slow and expensive
- Simple keyword/blacklist filters: fast but easily bypassed, high false-positive rate
- ML-based content classifiers: a practical middle ground — automatable, adaptable via retraining
- *Speaker notes:* Position this project against the two extremes (fully manual vs. naive rule-based).

### Slide 6 — Literature Survey (Papers 1–3)
- **Paper 1** — Ahmed, Traore & Saad (2017/2018): N-gram + TF-IDF, 6 ML classifiers, 92% accuracy on their ISOT dataset
- **Paper 2** — Shu, Sliva, Wang, Tang & Liu (2017): survey — content vs. social-context detection taxonomy
- **Paper 3** — Pérez-Rosas, Kleinberg, Lefevre & Mihalcea (2018): crowdsourced datasets, up to 76% accuracy
- *Speaker notes:* Emphasize the accuracy gap between Paper 1 (92%) and Paper 3 (76%) — it's dataset-dependent, not a fixed ceiling on the approach.

### Slide 7 — Literature Survey (Papers 4–5)
- **Paper 4** — Wang (2017), LIAR dataset: ~12.8K short political statements, fine-grained 6-way labels
- **Paper 5** — Zhou & Zafarani (2020) survey: 4-perspective taxonomy — knowledge, style, propagation, source
- *Speaker notes:* Use Paper 5's taxonomy to explicitly state: "our project sits in the style/content-based category."

### Slide 8 — Research Gap
- Reported accuracy varies heavily by dataset/domain (92% vs. 76% across papers)
- Most practical student-scale projects use content-only signals, ignoring propagation/source-credibility
- **Our contribution (Review 1 scope):** a clean, reproducible content-based baseline with transparent, non-fabricated evaluation on a clearly documented dataset
- *Speaker notes:* Be honest here — this is a baseline/reproduction-quality contribution, appropriate for Review 1, not a novel algorithm.

### Slide 9 — Dataset
- Fake and Real News Dataset (Kaggle, built on the ISOT Fake News Dataset, Univ. of Victoria)
- Columns: title, text, subject, date, label
- [Insert actual row counts / class balance from your `results/eda_summary.json` here once generated]
- *Speaker notes:* State clearly that these numbers come from your own EDA run, not from the paper.

### Slide 10 — Dataset Justification + EDA
- Justification: free, widely used, binary-labeled, large enough for TF-IDF + classical ML
- *Chart:* `results/class_distribution.png` (Fake vs Real counts)
- *Chart:* `results/article_length_distribution.png`
- *Chart:* `results/top_words_FAKE.png` and `top_words_REAL.png`
- *Speaker notes:* Point out any class imbalance and how the stratified split (Slide 12) handles it.

### Slide 11 — Proposed Methodology
- NLP preprocessing: lowercase → remove HTML/URLs → remove punctuation → tokenize → remove stopwords → lemmatize
- Feature extraction: TF-IDF, unigrams + bigrams, max_features=5000, on title+text combined
- Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- *Speaker notes:* Briefly explain *why* each preprocessing step is used (see `src/preprocessing.py` docstring) — examiners often ask "why lemmatization and not stemming?"

### Slide 12 — System Architecture / Workflow
- *Diagram:* the workflow chain from `docs/workflow_diagram.md` (Raw News → ... → Streamlit demo)
- *Speaker notes:* Walk through it top to bottom, one box per objective from Slide 4.

### Slide 13 — Technology Stack
- Python 3, pandas, numpy
- scikit-learn (TF-IDF, models, metrics)
- NLTK (tokenization, stopwords, lemmatization)
- matplotlib / seaborn (EDA, confusion matrices)
- joblib (model serialization)
- Streamlit (demo UI)
- *Speaker notes:* Mention these are all open-source, standard, well-documented libraries — a deliberate choice for reproducibility.

### Slide 14 — Implementation Progress
- Table (reuse from README §"Review 1 Implementation Status")
- Preprocessing, EDA, TF-IDF, all 3 models, evaluation, prediction function: **completed**
- Streamlit demo: **completed / in progress** (state your actual status honestly)
- Advanced model (LSTM/BERT): **future work**, not part of Review 1
- *Speaker notes:* This slide is your "~50% or more implementation" proof — be ready to demo live or show screenshots.

### Slide 15 — Initial Results + Confusion Matrix
- Comparison table: Model | Accuracy | Precision | Recall | F1 (from `results/model_comparison.csv` — real numbers, not invented)
- *Chart:* best model's confusion matrix PNG from `results/`
- State which model won and why (highest F1 — explain the precision/recall tradeoff rationale)
- *Speaker notes:* If the dataset isn't loaded yet at review time, say so explicitly and show the pipeline running instead of showing fake numbers.

### Slide 16 — Demo + Future Work
- Live demo (or screenshot) of the Streamlit app: paste a headline → REAL/FAKE + confidence
- Future work: Word2Vec+LSTM or DistilBERT (Section 13 of original brief), cross-domain evaluation, explainability (SHAP/LIME), multilingual support
- Thank you / Q&A
- *Speaker notes:* End on a clear "what's done vs. what's next" summary so examiners see scope was managed deliberately.
