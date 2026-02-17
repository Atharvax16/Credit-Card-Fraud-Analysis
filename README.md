# Credit Card Fraud Detection — End-to-End ML Analysis

This project is a hands-on exploration of building a **Credit Card Fraud Detection system** on a large real-world, highly imbalanced dataset.

The goal is not just training a model — but understanding:
- What features actually matter
- How to handle high-cardinality categorical data
- How to prevent leakage-like signals
- Why class imbalance breaks normal accuracy metrics
- How baseline statistical models behave in real data

This notebook documents the entire reasoning process and experiments step-by-step.

---

## Dataset Overview

The dataset contains transaction-level records with information about:

- Card holder details
- Merchant information
- Transaction timestamp
- Transaction amount
- Location coordinates
- Demographics
- Fraud label (`is_fraud`)

Approx dataset size: **~1.29 Million transactions**

Target:
is_fraud
0 → Legitimate
1 → Fraudulent


This is a **highly imbalanced dataset** (fraud << non-fraud), making accuracy misleading and recall/precision more important.

---

## Project Pipeline

### 1. Data Cleaning

Removed non-useful identifier columns:
Unnamed: 0
first
last
street
trans_num
trans_date_trans_time
dob


Reason:
These act as IDs or personal identifiers and do not help generalizable learning.

---

### 2. Feature Engineering

Created behavior-based features from timestamps:

- `trans_hour`
- `trans_dayofweek`
- `trans_month`
- `is_night`
- `age`

Why?
Fraud patterns are behavioral. Time-based features capture:
- Late night fraud
- Weekend activity spikes
- Seasonal anomalies

---

### 3. High Cardinality Analysis

Checked number of unique values in each column.

Some columns had extremely large categories:

| Feature | Unique Values |
|-------|------|
| merchant | ~693 |
| city | ~894 |
| job | ~494 |
| cc_num | ~983 |

Using one-hot encoding here would explode dimensions → bad generalization.

---

### 4. Target Encoding (K-Fold + Smoothing)

Instead of one-hot encoding high-cardinality columns, applied **Target Encoding**:

Encoded columns:
- merchant
- city
- job

Technique used:
- K-Fold target encoding to avoid leakage
- Smoothing to prevent rare category overfitting

This converts categories into probability-based numerical signals.

Example idea:
merchant_te ≈ probability that this merchant is fraud-prone


---

### 5. Removing Location-Heavy Columns

Dropped:
lat
long
merch_lat
merch_long
zip


Reason:
- Location columns often behave like near-duplicate identity signals
- May unintentionally leak pattern memorization
- Improves model generalization

---

### 6. Train-Test Split

Used stratified split:
train_test_split(..., stratify=y, test_size=0.2, random_state=42)

Important because fraud class is extremely rare.


## How to Run

### Clone repository
git clone https://github.com/Atharvax16/Credit-Card-Fraud-Analysis.git
cd Credit-Card-Fraud-Analysis

### Create Environment
python -m venv .venv

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

### Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

### Run notebook
jupyter notebook

### Open:
Credit_Fraud.ipynb

## Repository Structure:
Credit-Card-Fraud-Analysis/

│

├── data/

│   └── dataset files

│
├── Credit_Fraud.ipynb

└── README.md

## Learning Outcomes
### Through this project:
Understood why accuracy fails in imbalanced ML problems
Learned importance of class weighting
Applied K-Fold target encoding correctly
Identified leakage-like signals
Built a statistically meaningful baseline
This project focuses on understanding model behavior before chasing high accuracy.

## Author
### Atharva Kocharekar
GitHub: https://github.com/Atharvax16
