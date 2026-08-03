# E-Commerce Fraud Detection Dashboard

An AI-powered e-commerce scam detection system built during a CCRI summer internship (June–August 2026). The project combines machine learning, unsupervised clustering, and a multi-page Streamlit dashboard to detect three distinct categories of e-commerce fraud: transaction fraud, phishing websites, and AI-generated fake reviews.

**Live app:** https://ecommerce-scam-detection-daur7q6ebnaqedg3wkqyok.streamlit.app/

## Project Scope

The original brief called for a broad scam detection tool covering fake listings, counterfeit sellers, phishing scams, fake reviews, payment fraud, and fraudulent ads. Given dataset availability constraints (see Limitations below), this project focuses on three independently trained, dataset-supported modules rather than attempting shallow coverage of all categories:

1. **Transaction Fraud Detection** — the core module, combining supervised and unsupervised learning
2. **Website Risk Checker** — phishing URL detection
3. **Review Authenticity Checker** — AI-generated review detection

Each module uses its own dataset and model, unified under a single dashboard.

## Setup

```bash
cd D:\ecommerce-scam-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

Model and dataset files are hosted via GitHub Releases (due to GitHub's 100MB file limit) and download automatically on first run via `src/data_loader.py`.

## Run the Notebooks

```bash
jupyter notebook
```

## Project Structure

```
├── app.py                          # Streamlit dashboard (all pages)
├── data/                           # Datasets (gitignored, auto-downloaded)
├── models/                         # Trained models (gitignored, auto-downloaded)
├── notebooks/
│   ├── 01_load_and_explore_data.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_test_fraud_score.ipynb
│   ├── 06_transaction_archetypes.ipynb
│   ├── 07_phishing_module.ipynb
│   └── 08_fake_review_module.ipynb
├── src/
│   ├── data_loader.py              # Auto-downloads models/data from GitHub Releases
│   ├── fraud_scorer.py             # XGBoost transaction fraud model wrapper
│   ├── fraud_score.py              # FraudScoreEngine: composite scoring (ML + category + location + archetype)
│   ├── product_analyzer.py         # Category-level fraud rate analysis
│   ├── seller_analyzer.py          # Location-level fraud rate analysis
│   ├── phishing_checker.py         # Phishing URL classifier + live feature extraction
│   ├── review_checker.py           # Review authenticity classifier
│   └── cluster_risk_map.json       # K-Means cluster → fraud rate mapping
└── requirements.txt
```

## Module 1: Transaction Fraud Detection

**Approach:** a composite score fusing four signals:
- XGBoost classifier (60% weight) — trained on ~1.47M transaction records
- Product category fraud rate (15% weight)
- Customer location fraud rate (10% weight)
- K-Means transaction archetype cluster risk (15% weight)

**Key technical contribution:** the dataset has no repeating customer, IP, or address identifiers, making traditional entity-level behavioral profiling impossible. Instead of abandoning behavioral analysis, transaction-level archetype clustering was developed as a substitute: six self-contained engineered features (price deviation, address mismatch, odd-hour flag, new-account flag, quantity deviation, amount-per-account-age) are clustered via K-Means, and each cluster's historical fraud rate becomes a risk signal.

**Performance (on a 50,000-row held-out sample):**
| Metric | XGBoost Only | Composite Fusion (threshold=50) |
|---|---|---|
| Accuracy | 0.81 | 0.94 |
| ROC-AUC | 0.84 | 0.85 |
| Recall | 0.71 | 0.44 |
| Precision | 0.17 | 0.42 |

The Fraud-tier threshold was empirically recalibrated from an initial 70 to 50 after testing across a range of cutoffs, improving F1 score.

**Dataset:** Fraudulent E-Commerce Transaction Data (~1.47M rows)

## Module 2: Website Risk Checker

**Approach:** Random Forest classifier trained on 19 URL-structural features (length, character composition, subdomain count, HTTPS usage, special character ratios), computed live from any pasted URL.

**Performance:** 99.7% accuracy, 0.998 ROC-AUC on held-out test data.

**Known limitation:** training data underrepresents non-Western domains (only 82 of 235,795 URLs contained `.co.id`), causing legitimate regional websites (e.g., Indonesian banking/corporate sites) to be misclassified as phishing. A feature engineering bug affecting compound TLD handling (e.g., `.co.id`, `.co.uk`) was identified and fixed during testing; the remaining misclassification is attributed to dataset representativeness, not code defects.

**Dataset:** PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024, *Computers & Security*)

## Module 3: Review Authenticity Checker

**Approach:** TF-IDF vectorization (5,000 features, unigrams and bigrams) with a Logistic Regression classifier.

**Performance:** 87.8% accuracy on held-out test data.

**Known limitation:** the training dataset's "AI-generated" examples reflect an older, more repetitive text-generation style (e.g., repeated template phrases like "the only problem is"). Testing against modern ChatGPT-generated reviews showed the model consistently misclassifies them as genuine, since it learned to detect a specific historical generation pattern rather than AI-generated text in general. This reflects a broader, ongoing challenge in AI-content detection research.

**Dataset:** Fake Reviews Dataset (Kaggle, Mexwell) — 40,432 reviews, balanced 50/50 human-written vs. computer-generated

## Dashboard Pages

- **Dashboard Overview** — summary statistics
- **Check a Transaction** — single-transaction fraud scoring
- **Analytics** — fraud rate breakdowns by category, payment method, and location
- **Archetype Clusters** — cluster-level fraud concentration and PCA visualization
- **Batch Alerts** — CSV upload for bulk transaction scoring with downloadable results
- **Website Risk Checker** — single-URL phishing risk scoring
- **Review Authenticity Checker** — single-review AI-generation likelihood scoring
- **Info** — project and module documentation

## Key Technical Decisions & Lessons

- **Dataset structure shapes architecture:** no repeating entity identifiers ruled out traditional seller/customer profiling, driving the pivot to transaction-level archetype clustering.
- **Feature consistency is critical:** an early bug trained the phishing model on the dataset's precomputed features but predicted using a reconstructed feature function, causing systematic misclassification (e.g., flagging google.com as phishing). Fixed by using one single feature-extraction function for both training and inference.
- **O(1) lookups over linear scans:** replacing an `x in encoder.classes_` array scan with a precomputed dictionary lookup reduced per-row encoding time from ~28 seconds to ~0.2 seconds.
- **Path resolution must be absolute:** `Path(__file__).resolve().parent.parent` is used throughout so modules work correctly whether called from a notebook or from the deployed app's root context.
- **Threshold assumptions should be tested, not guessed:** risk-tier cutoffs were empirically validated against precision/recall trade-offs rather than left at initial round-number estimates.

## Deployment

Large files (models, datasets) exceed GitHub's 100MB limit and are hosted via GitHub Releases, auto-downloaded on app startup by `src/data_loader.py`. The app is deployed on Streamlit Cloud.

## Limitations & Future Work

- **Scope:** the original brief covered fake listings, counterfeit sellers, and fraudulent ads, which are not addressed here due to lack of suitable public datasets with the necessary granularity.
- **Regional bias (Website Risk Checker):** would require a larger, more geographically diverse training set to resolve.
- **Detection generation gap (Review Authenticity Checker):** would require retraining on more recent LLM-generated samples to keep pace with advancing generation models.
- **Scalability:** the current architecture uses local model files and static datasets. A production deployment would benefit from a managed ML platform (e.g., Google Cloud Vertex AI) for automated retraining pipelines and larger-scale data ingestion.

## Datasets

- Fraudulent E-Commerce Transaction Data
- PhiUSIIL Phishing URL Dataset — Prasad, A. & Chandra, S. (2024). *Computers & Security*
- Fake Reviews Dataset — Kaggle (Mexwell)
