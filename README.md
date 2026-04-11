# 💳 Credit Card Fraud Detection

## 📌 Project Overview
This project builds a fraud detection system using machine learning and deep learning techniques to identify fraudulent credit card transactions.

Fraud detection is a highly imbalanced classification problem. This project handles it using preprocessing, sampling techniques, model tuning, and explainability tools.

---

## 🎯 Objectives
- Detect fraudulent transactions with high recall
- Minimize false negatives
- Handle class imbalance effectively
- Compare multiple models

---

## 📂 Dataset
- Credit Card Fraud Detection Dataset
- Features:
  - V1–V28 (anonymized)
  - Time, Amount
  - Class (0 = Legit, 1 = Fraud)

---

## ⚙️ Workflow

### 1. Exploratory Data Analysis
- Class distribution
- Transaction analysis
- Correlation analysis

### 2. Data Preprocessing
- Feature engineering (Time → Hour)
- Feature selection
- Train-test split

### 3. Handling Imbalance
- Undersampling
- SMOTE
- Class weights

### 4. Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Isolation Forest
- LightGBM
- Neural Network (MLP)

### 5. Optimization
- Optuna tuning
- Threshold tuning
- Cost-sensitive learning

### 6. Evaluation
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

### 7. Explainability
- SHAP for feature importance

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn
- LightGBM
- PyTorch
- Optuna
- SHAP

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Atharvax16/Credit-Card-Fraud-Analysis.git
cd Credit-Card-Fraud-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
jupyter notebook
```

---

## 📁 Structure
```
project/
│── data/
│── notebooks/
│── README.md
│── requirements.txt
```

---

## 📌 Key Points
- Class imbalance is a major challenge
- Recall is more important than accuracy
- SHAP helps in model interpretability

---

## 🔮 Future Work
- Model deployment (Flask/FastAPI)
- Real-time detection
- Deep learning improvements

---

## 👨‍💻 Author
Atharva Kocharekar
