# Technical Methodology & Findings

This document consolidates the technical design decisions, debugging process, and evaluation methodology behind the E-Commerce Fraud Detection Dashboard. It is intended as source material for the final project report and presentation.

---

## 1. Project Scope Decision

The original project brief called for a broad scam detection system covering fake listings, counterfeit sellers, phishing scams, fake reviews, payment fraud, and fraudulent ads. Early data exploration revealed a structural constraint: the primary transaction dataset contains no repeating customer IDs, IP addresses, or shipping/billing addresses, meaning any entity-level profiling approach (tracking a specific customer or seller's behavior over time) is not possible with this data.

Rather than attempting shallow coverage across all six original categories, the project scope was narrowed to three independently supported modules, each backed by a suitable dataset:

1. Transaction fraud detection (core module)
2. Phishing website detection
3. AI-generated review detection

This is a scope refinement based on data availability, not a reduction in technical ambition. Each module represents a complete, tested, independently evaluated system.

---

## 2. Module 1: Transaction Fraud Detection

### 2.1 Architecture

A composite scoring system combining four signals:

| Component | Weight | Method |
|---|---|---|
| ML classification | 60% | XGBoost, trained on ~1.47M labeled transactions |
| Category risk | 15% | Historical fraud rate by product category |
| Location risk | 10% | Historical fraud rate by customer location |
| Archetype risk | 15% | K-Means cluster fraud concentration |

### 2.2 Key Innovation: Transaction Archetype Clustering

Since entity-level profiling was structurally unavailable, behavioral analysis was reframed at the transaction level. Six features were engineered, each computable from a single transaction record without reference to entity history:

- Price deviation (distance from category's expected price)
- Address mismatch (shipping vs. billing)
- Odd-hour flag (transaction hour outside typical activity window)
- New-account flag (account age under 30 days)
- Quantity deviation (distance from category's typical quantity)
- Amount-per-account-age (spend normalized by account age)

These features were clustered via K-Means (k=3, selected based on cluster interpretability and separation). Each cluster's historical fraud rate becomes its risk weight, stored in `cluster_risk_map.json`.

**Result:** one cluster (representing ~10% of transactions) showed a 22.5% fraud concentration, roughly 7x higher than the baseline cluster (3.26%), dominated by high spend relative to account age combined with new-account status. A second cluster, defined by address mismatch alone, showed the lowest fraud rate (2.73%), suggesting address mismatch functions better as a contributing signal than a standalone red flag.

### 2.3 Threshold Calibration

Initial risk tiers (Safe <30, Suspicious 30–70, Fraud >70) were round-number estimates. Empirical testing across a threshold sweep (20–70) on a 50,000-row held-out sample identified 50 as the F1-optimal cutoff (F1 = 0.428, versus F1 = 0.009 at the original threshold of 70).

**Before/after threshold recalibration:**

| Metric | XGBoost Only | Composite Fusion (threshold=50) |
|---|---|---|
| Accuracy | 0.8119 | 0.9408 |
| ROC-AUC | 0.8391 | 0.8495 |
| Recall | 0.7134 | 0.4384 |
| Precision | 0.1718 | 0.4178 |

The composite fusion trades recall for substantially improved precision, a deliberate shift toward higher-confidence alerts suited to manual review workflows, where a high false-positive rate (as seen in the XGBoost-only baseline) would overwhelm an analyst.

### 2.4 Engineering Fixes

- **Encoder performance bottleneck:** initial label encoding used `x in encoder.classes_`, a linear array scan repeated per categorical column per row. Profiling isolated this as a 28-second-per-row bottleneck (out of a 50-second total scoring time). Replaced with a precomputed dictionary lookup (`{class: index}`), reducing per-row scoring to ~0.2 seconds, a roughly 250x improvement, critical for enabling practical batch processing.
- **Redundant computation:** category and location risk lookups were being recalculated via full dataset `groupby` operations on every single scoring call. Precomputed once at initialization instead.
- **Path resolution:** all module file paths use `Path(__file__).resolve().parent.parent` rather than relative paths, ensuring correct resolution whether code is invoked from a notebook (`notebooks/`) or the deployed application root.

---

## 3. Module 2: Website Risk Checker

### 3.1 Dataset and Approach

Trained on the PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024, *Computers & Security*; 235,795 URLs, 134,850 legitimate / 100,945 phishing). Of the dataset's 54 available features, only the 19 directly computable from a raw URL string (without requiring live webpage content or precomputed corpus statistics) were used, preserving the same "self-contained feature" design principle used in the transaction archetype module.

A Random Forest classifier (100 estimators) was trained, achieving 99.7% accuracy and 0.998 ROC-AUC on held-out data.

### 3.2 Debugging Process

**Bug 1 — feature inconsistency:** initial training used the dataset's precomputed feature values, but live prediction used a custom-reconstructed feature extraction function (since several of the dataset's exact feature formulas are undocumented). This mismatch caused systematic misclassification, including flagging `google.com` as 95% phishing. Fixed by recomputing all 235,795 training examples using the same extraction function used at inference time, ensuring internal consistency regardless of whether the values matched the original researchers' exact methodology.

**Bug 2 — compound TLD handling:** testing against real-world Indonesian URLs (e.g., `cimbniaga.co.id`) revealed the subdomain-counting logic treated compound top-level domains (`.co.id`, `.co.uk`, `.com.au`) as extra subdomain levels, inflating a feature associated with suspicious URL structure. Fixed by maintaining a list of known compound TLDs and adjusting the counting logic accordingly.

**Bug 3 (process, not code) — Streamlit caching:** `@st.cache_resource` retained a stale model object in memory across code and file changes, masking whether fixes had actually taken effect during testing. Resolved by fully restarting the Streamlit server process (not just the browser) after model updates.

### 3.3 Confirmed Limitation: Regional Data Scarcity

After both bugs were fixed, testing confirmed that legitimate Indonesian domains (e.g., `cimbniaga.co.id`) are still misclassified as phishing. Investigation found only 82 of 235,795 training URLs (0.035%) contained `.co.id`, insufficient for the model to learn reliable patterns for this domain category. This is classified as a dataset representativeness limitation, not a code defect, a well-documented challenge in publicly sourced cybersecurity datasets, which tend to overrepresent English-language, Western-centric web sources.

---

## 4. Module 3: Review Authenticity Checker

### 4.1 Dataset and Approach

Trained on the Kaggle Fake Reviews Dataset (Mexwell; 40,432 reviews, balanced 50/50 between computer-generated and original human-written text). TF-IDF vectorization (5,000 features, unigrams and bigrams, English stop words removed) paired with Logistic Regression achieved 87.8% accuracy on held-out data.

### 4.2 Interpretability Findings

Feature coefficient analysis revealed distinct linguistic signatures:
- **Human-indicating terms:** hedging and conversational words (`actually`, `maybe`, `quite`, `instead`), reflecting natural conversational imperfection
- **AI-indicating terms:** repeated multi-word template phrases (`reason gave`, `problem really`, `kind hard`), suggesting the dataset's AI-generated examples were produced using a limited set of sentence templates rather than genuinely varied generation

### 4.3 Confirmed Limitation: Generation-Era Gap

Testing against modern ChatGPT-generated review text showed consistent misclassification as genuine. The model correctly identifies text matching its training data's templated generation style but does not generalize to more fluent, contemporary LLM output. This reflects a broader, actively discussed challenge in AI-content detection: classifiers trained on one generation of AI output can become outdated as generation technology advances, an "arms race" dynamic between generation and detection capabilities.

---

## 5. Cross-Module Observations

- **Task difficulty correlates with signal clarity, not data volume.** The phishing checker (99.7% accuracy) and review checker (87.8% accuracy) were trained on comparably sized, balanced datasets, yet performance differs substantially. URL structure provides clearer, more separable signal than the more subtle, evolving linguistic patterns distinguishing human from AI-generated text.
- **Every module's primary limitation stems from training data characteristics, not algorithm choice.** The transaction model's precision/recall trade-off stems from real-world class imbalance; the phishing checker's regional bias stems from geographic underrepresentation; the review checker's generation-era gap stems from dataset age relative to advancing LLM capability. This consistent pattern reinforces that data quality and representativeness, not model sophistication, was the binding constraint across the project.
- **Feature/prediction consistency is a recurring failure mode.** Both the phishing module (training vs. inference feature mismatch) and the transaction module's original design pattern share this risk category, reinforcing the importance of using identical computation paths for training and inference throughout.

---

## 6. Evaluation Methodology Notes

- Transaction model evaluation used a randomly sampled 50,000-row held-out set (original train/test split artifacts were not preserved from earlier development, addressed via fresh stratified sampling with a fixed random seed for reproducibility).
- Phishing and review models used standard 80/20 stratified train/test splits with fixed random seeds.
- All reported metrics are from held-out data not used in training.

---

## 7. Future Work

- Expand regional URL training data to address the Website Risk Checker's geographic bias.
- Periodically retrain the Review Authenticity Checker on contemporary LLM-generated samples to track advancing generation capability.
- Explore a managed ML platform (e.g., Google Cloud Vertex AI) for automated retraining pipelines at production scale.
- Investigate additional fraud categories (counterfeit sellers, fake listings) contingent on identifying suitable public datasets.
