# User Guide — E-Commerce Fraud Detection Dashboard

This guide walks through each page of the dashboard from the perspective of a marketplace analyst using the tool to screen transactions, websites, and reviews for fraud.

## Getting Started

Open the dashboard (locally via `streamlit run app.py`, or the live Streamlit Cloud link). On first load, models and datasets download automatically, this may take a few moments the first time. Navigate between pages using the sidebar on the left.

---

## Dashboard Overview

The landing page. Shows three headline statistics for the transaction dataset:
- **Total Transactions** — size of the reference dataset
- **Fraud Rate** — overall percentage of transactions labeled fraudulent
- **Fraud Cases** — raw count of fraudulent transactions

Use this page to confirm the app has loaded correctly before using other features.

---

## Check a Transaction

Use this page to evaluate a single transaction manually.

**How to use it:**
1. Fill in the form: transaction amount, product category, quantity, customer age, location, account age, transaction hour, shipping/billing address, payment method, and device used.
2. Click **Check Transaction**.
3. Review the result:
   - A colored label (🟢 Safe / 🟡 Suspicious / 🔴 Fraud)
   - A breakdown of four risk components: ML Score, Category Risk, Location Risk, and Archetype Risk
   - The Final Composite Score (0–100), which combines all four

**Tip:** to see a high-risk example, set Account Age to 0, use a high transaction amount, set the hour to an unusual time (e.g., 3 AM), and use different shipping/billing addresses.

---

## Analytics

Read-only charts summarizing fraud patterns across the full dataset:
- **Fraud Rate by Product Category** — which product types see the most fraud
- **Fraud Rate by Payment Method** — which payment methods are riskiest
- **Top 15 Highest-Risk Locations** — customer locations with the highest fraud concentration (filtered to locations with at least 10 transactions, to avoid misleading results from very small samples)

Use this page to understand broad fraud trends before drilling into individual cases.

---

## Archetype Clusters

Shows the results of unsupervised behavioral clustering, a novel layer that groups transactions by shared risk-relevant characteristics rather than by customer identity (since the dataset has no repeating customers).

**What you'll see:**
- **Cluster Summary table** — transaction count and fraud rate per cluster
- **Dominant Feature Signature table** — average value of each of the six engineered risk features per cluster, useful for understanding what defines each cluster
- **2D Projection chart** — a visual map of how transactions group together, colored by cluster and marked by actual fraud outcome

Use this page to understand *why* certain transaction patterns are considered risky, beyond just the final score.

---

## Batch Alerts

Use this page to screen many transactions at once, ideal for reviewing a batch of pending orders.

**How to use it:**
1. Prepare a CSV file with these columns: `Transaction Amount, Product Category, Quantity, Customer Age, Customer Location, Account Age Days, Transaction Hour, Shipping Address, Billing Address, Payment Method, Device Used`
2. Upload the file.
3. Wait for scoring to complete (a progress spinner shows while processing).
4. Review the summary counts (Safe / Suspicious / Fraud), the table of flagged transactions sorted by risk, and the full results table.
5. Click **Download Full Results as CSV** to export the scored data.

**Note:** files with missing required columns, blank values in required fields, or empty files are detected and flagged with clear error messages rather than crashing.

---

## Website Risk Checker

Use this page to check whether a URL shows signs of being a phishing site.

**How to use it:**
1. Paste a URL into the input field (include `https://` or `http://`).
2. Click **Check Website**.
3. Review the result: Legitimate or Phishing, with a confidence percentage.

**Important limitation:** this checker is trained primarily on Western/English-language URL patterns. It may be less reliable for regional domains (e.g., Indonesian `.co.id` sites), which can be misclassified due to underrepresentation in the training data. Always apply additional judgment for sites outside common Western patterns.

---

## Review Authenticity Checker

Use this page to check whether a product review shows signs of being AI-generated rather than written by a genuine customer.

**How to use it:**
1. Paste review text into the text box.
2. Click **Check Review**.
3. Review the result: Likely Genuine or Likely AI-Generated, with a confidence percentage.

**Important limitations:**
- Very short review text (under ~20 characters) produces less reliable predictions, since the model relies on sentence-level patterns.
- This checker was trained on an older style of AI-generated text and is less effective at catching modern, more fluent AI writing (such as current-generation ChatGPT output). It reliably catches repetitive, templated phrasing but should not be treated as a comprehensive AI-detection tool.

---

## Info

A summary page describing all three detection modules, their underlying models, and their data sources. Useful as a quick reference for understanding the system's overall design without digging into the code.

---

## General Notes

- All three detection modules are **independent**, each trained on its own dataset for its own specific fraud type. They do not share data or influence each other's scores.
- Every module's known limitations are documented directly in its page (via the caption notes) and in this guide, in the interest of transparency about what the system can and cannot reliably catch.
