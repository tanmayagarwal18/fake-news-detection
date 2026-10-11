# Literature Survey — Fake News Detection Using NLP and Machine Learning

Five genuine, verifiable papers relevant to this project. Citations
checked against publisher/repository sources (Wiley, ACM, ACL
Anthology, arXiv) before inclusion — no fabricated papers, authors, or
metrics.

---

### Paper 1

**Title:** Detecting Opinion Spams and Fake News Using Text Classification
**Authors:** Hadeer Ahmed, Issa Traore, Sherif Saad
**Year:** 2018 (journal version); an earlier conference version appeared in 2017 (ISDDC, Springer LNCS vol. 10618)
**Venue:** *Security and Privacy*, Wiley, Vol. 1, Issue 1, e9
**Dataset:** The authors' own **ISOT Fake News Dataset** (collected by the University of Victoria ISOT Research Lab) plus other existing opinion-spam datasets.
**Methodology:** N-gram feature extraction (up to a chosen n) combined with TF and TF-IDF weighting, feeding six different ML classifiers.
**Model/Algorithm:** Compared six classifiers, including Naive Bayes, SVM, Logistic Regression, and ensemble-style linear models over n-gram/TF-IDF features.
The proposed model achieved 92% accuracy on a dataset of 12,600 fake and 12,600 legitimate news articles.
**Limitation:** Purely content/linguistic-feature based; does not use social-context signals (who shared it, propagation pattern), and the dataset is domain/time-limited (mostly U.S. political news).
**Relevance to our project:** This is the direct source of the ISOT dataset used as our recommended dataset, and its n-gram/TF-IDF + classical-ML pipeline is essentially the architecture this project implements.

---

### Paper 2

**Title:** Fake News Detection on Social Media: A Data Mining Perspective
**Authors:** Kai Shu, Amy Sliva, Suhang Wang, Jiliang Tang, Huan Liu
**Year:** 2017
**Venue:** ACM SIGKDD Explorations Newsletter, Vol. 19, No. 1 (also arXiv:1708.01967)
**Dataset:** Survey paper — reviews multiple existing datasets/benchmarks rather than introducing one of its own.
**Methodology:** Structures the problem from a data-mining perspective: characterizes fake news using both psychology/social theories and computational features (news content features + social-context features).
**Model/Algorithm:** Surveys content-based (linguistic, style) and social-context-based (propagation, user credibility) detection algorithms rather than proposing a single model.
**Important result:** Provides a comprehensive review of fake news detection on social media, covering psychology/social theory characterizations, data-mining algorithms, evaluation metrics, representative datasets, and future research directions.
**Limitation:** Being a survey, it does not report a single benchmark accuracy; practical performance depends on which of the surveyed approaches is implemented.
**Relevance to our project:** Frames why content-only (text) detection — the approach this project uses — is one valid piece of a larger detection space, and motivates our "Research Gap" and "Future Work" sections (adding social-context signals later).

---

### Paper 3

**Title:** Automatic Detection of Fake News
**Authors:** Verónica Pérez-Rosas, Bennett Kleinberg, Alexandra Lefevre, Rada Mihalcea
**Year:** 2018 (COLING 2018 proceedings; also circulated as arXiv:1708.07104 in 2017)
**Venue:** Proceedings of the 27th International Conference on Computational Linguistics (COLING), pp. 3391-3401, ACL Anthology
**Dataset:** The authors introduce two novel datasets covering seven different news domains, built via crowdsourced (Amazon Mechanical Turk) fake-article writing paired with matching legitimate articles.
**Methodology:** Linguistic feature engineering (n-grams, syntax, psycholinguistic/LIWC-style features) fed into classical ML classifiers; also compares automatic classification against human ability to spot fake news.
**Model/Algorithm:** Classical supervised ML classifiers over linguistic feature sets.
**Important result:** The learned models reach accuracies of up to 76% on the introduced datasets.
**Limitation:** Crowdsourced fake articles may not fully mimic the linguistic style of real-world, professionally produced disinformation; 76% accuracy is noticeably lower than results reported on the more homogeneous ISOT-style datasets, highlighting how much dataset choice affects reported performance.
**Relevance to our project:** Useful counterpoint in our Research Gap section — shows that reported accuracy is highly dataset-dependent, so our own results should be interpreted relative to the specific dataset used, not as a universal fake-news-detection accuracy.

---

### Paper 4

**Title:** "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection
**Authors:** William Yang Wang
**Year:** 2017
**Venue:** arXiv:1705.00648 (widely cited benchmark paper; commonly referenced at ACL-affiliated venues)
**Dataset:** Introduces the **LIAR** dataset — about 12.8K human-labeled short political statements from PolitiFact, each with a fine-grained truthfulness label (not just binary) plus speaker metadata.
**Methodology:** Benchmarks statement classification using both text-only models and hybrid models that also use speaker-level metadata (party, job, history of past statement ratings).
**Model/Algorithm:** Classical ML and neural baselines (e.g. CNN-based models) compared on the fine-grained (6-class) and text+metadata settings.
**Limitation:** Statements are short (closer to claims/headlines than full articles) and the 6-way fine-grained label scheme is harder than binary real/fake, so results aren't directly comparable to binary-classification papers like Papers 1 and 3.
**Relevance to our project:** Justifies our choice to work with a binary-labeled, full-article dataset (ISOT) for a first academic project, while noting LIAR as a natural harder benchmark for future work/extension.

---

### Paper 5

**Title:** A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities
**Authors:** Xinyi Zhou, Reza Zafarani
**Year:** 2020
**Venue:** ACM Computing Surveys, Vol. 53, No. 5, Article 109
**Dataset:** Survey paper — reviews the fake-news-detection literature broadly rather than using a single dataset.
**Methodology:** Reviews and evaluates detection methods from four perspectives: the false knowledge a story carries, its writing style, its propagation pattern, and the credibility of its source, and connects these to grounding theories from multiple disciplines.
**Model/Algorithm:** N/A (survey/taxonomy of methods, not a single proposed model).
**Important result:** Organizes the field into a four-perspective taxonomy (knowledge-based, style-based, propagation-based, source-based) that is widely used in later papers to categorize new methods.
**Limitation:** As a survey, offers no new empirical benchmark; the taxonomy is broad, so mapping a specific method (like ours) onto it requires some interpretation.
**Relevance to our project:** This project's TF-IDF + classical ML pipeline sits squarely in the **style-based / content-based** category of Zhou & Zafarani's taxonomy — useful vocabulary for explaining the project's positioning in the viva.

---

## Comparison Table

| Paper | Dataset | Method | Result | Limitation |
|---|---|---|---|---|
| 1. Ahmed, Traore & Saad (2017/2018) | ISOT Fake News Dataset (own) | N-gram + TF-IDF, 6 ML classifiers | 92% accuracy (12,600 fake / 12,600 real) | Content-only; domain/time-limited |
| 2. Shu, Sliva, Wang, Tang & Liu (2017) | Survey — multiple datasets reviewed | Data-mining taxonomy (content + social-context) | N/A (survey) | No single benchmark reported |
| 3. Pérez-Rosas, Kleinberg, Lefevre & Mihalcea (2018) | 2 new crowdsourced datasets, 7 domains | Linguistic features + classical ML | Up to 76% accuracy | Crowdsourced fakes may not match real disinformation style |
| 4. Wang (2017) — LIAR | LIAR: ~12.8K PolitiFact statements | Text + speaker metadata, ML/CNN baselines | Benchmark dataset (6-way fine-grained) | Short statements, not full articles |
| 5. Zhou & Zafarani (2020) | Survey — no single dataset | 4-perspective taxonomy (knowledge/style/propagation/source) | N/A (survey) | No new empirical results |

## Research Gap

Across these papers, two consistent gaps emerge that this project's
scope, and its future-work section, respond to:

1. **Dataset/domain dependence of reported accuracy.** Paper 1 reports
   92% on ISOT while Paper 3 reports only up to 76% on a different,
   crowdsourced dataset — the same general approach (content-based
   classical ML) performs very differently depending on the dataset's
   source homogeneity and article length. Our project addresses this by
   being explicit in the README about which dataset the reported
   numbers came from, rather than presenting a single accuracy figure
   as if it were a universal result.
2. **Content-only methods ignore propagation/social-context signals.**
   Papers 2 and 5 (the two surveys) both emphasize that style/content
   features are only one of several available signal types (alongside
   source credibility and propagation pattern). This project
   intentionally scopes itself to the content-only category for
   Review 1 (a feasible, well-understood baseline for a B.Tech
   timeline), and lists adding source/propagation features as future
   work rather than claiming to solve the full problem.
