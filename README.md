# Credit Card Fraud Analysis (End-to-End Notebook)

This repo is my hands-on walkthrough of building a **credit card fraud detection** pipeline on a large, highly-imbalanced dataset.  
Goal: go beyond “just training a model” and actually work through the messy parts — **feature engineering, high-cardinality categoricals, leakage-ish columns, imbalance handling**, and a strong baseline.

---

## What’s inside

- A complete analysis notebook: **`Credit_Fraud.ipynb`**
- Dataset placed under **`data/`**
- Work done so far includes:
  - Cleaning + dropping unnecessary identifiers
  - Feature engineering from timestamps + age
  - Handling **high-cardinality categorical** features using **Target Encoding (K-Fold + smoothing)**
  - One-hot encoding for low-card features (like `state`)
  - Correlation checks + dropping location features (`lat/long`, `merch_lat/merch_long`, `zip`) to reduce leakage / redundancy
  - Baseline model: **Logistic Regression** with **`class_weight="balanced"`**

---

## Dataset summary

The dataset contains transaction-level information with fields like:

- Transaction time: `trans_date_trans_time`, plus derived features
- Amount: `amt`
- Card + merchant identifiers: `cc_num`, `merchant`
- Category + demographics: `category`, `gender`, `job`, `city`, `state`, etc.
- Geo fields: `lat`, `long`, `merch_lat`, `merch_long`
- Target: **`is_fraud`** (0 = legit, 1 = fraud)

✅ Dataset size seen in notebook: **~1,296,675 rows** (large + imbalanced).

---

## Project workflow (what I did till now)

### 1) Imports & setup
Used:
- `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

### 2) Initial cleaning / column dropping
Dropped columns that are either pure identifiers or not useful for modeling in this stage:

###3) Feature engineering

Created extra behavioral/time-based signals (done before modeling):

trans_hour
trans_dayofweek
trans_month
is_night
age

These help the model learn “when fraud tends to happen” and “who/what profile is riskier”.

###4) High-cardinality check

Checked unique values per column to identify where one-hot encoding would explode:

Examples of high cardinality:
merchant (~693 unique)
city (~894)
job (~494)
cc_num (~983)
unix_time, merch_lat, merch_long etc. (very high)

```python
to_drop_cols = ['Unnamed: 0','first','last','street','trans_date_trans_time','dob']

###6) Correlation analysis + dropping location-heavy columns

After encoding + preprocessing, computed correlation matrix and then dropped:
lat, long, merch_lat, merch_long
zip
Reason (practical ML reason):

geo columns can introduce near-duplicate location signals

can behave like a shortcut / leakage-ish feature depending on how the dataset is constructed

also reduces noise + improves generalization
