Credit Card Fraud Detection (End-to-End ML Project)
📌 Overview

This project focuses on detecting fraudulent credit card transactions using machine learning and deep learning techniques. It covers the complete pipeline from data analysis and preprocessing to model building, evaluation, optimization, and interpretability.

The goal is to build highly accurate models that can effectively identify fraud while handling severe class imbalance, which is a key challenge in fraud detection.

📂 Dataset
Dataset used: Credit Card Fraud Detection Dataset
Contains anonymized transaction features (V1–V28), Time, Amount, and target variable Class
0 → Legitimate transaction
1 → Fraudulent transaction
⚙️ Project Workflow
1. 📊 Exploratory Data Analysis (EDA)
Checked missing values and data distribution
Visualized class imbalance (fraud vs legit)
Distribution analysis of:
Transaction amounts
Time-based patterns
Correlation analysis with fraud class
2. 🧹 Data Preprocessing
Feature engineering:
Converted Time → Hour
Feature selection:
Correlation-based selection
Mutual Information
Train-test split with stratification
3. ⚖️ Handling Class Imbalance
Undersampling (RandomUnderSampler)
Oversampling (SMOTE)
Class-weight balancing in models
4. 🤖 Models Implemented
Traditional ML Models
Logistic Regression
Random Forest
Gradient Boosting
Advanced Models
Isolation Forest (Anomaly Detection)
Neural Network (MLP using PyTorch)
LightGBM (with hyperparameter tuning)
5. 🔧 Model Optimization
Optuna Hyperparameter Tuning
Cost-sensitive learning
Penalizing false negatives more than false positives
Threshold tuning for better recall/precision balance
Probability calibration
6. 📈 Evaluation Metrics
Confusion Matrix
Precision, Recall, F1-score
ROC-AUC Score
Precision-Recall Curve
7. 🔍 Model Interpretability
SHAP (SHapley Additive exPlanations):
Feature importance
Individual prediction explanations (waterfall plots)
8. 📊 Final Comparison
Compared all models side-by-side
Visualized:
Model performance metrics
ROC curves
Identified best-performing model for fraud detection
🛠️ Tech Stack
Languages & Libraries
Python
Pandas, NumPy
Matplotlib, Seaborn
Machine Learning
Scikit-learn
Imbalanced-learn (SMOTE, undersampling)
Advanced ML / DL
LightGBM
PyTorch (MLP model)
Optimization & Explainability
Optuna
SHAP
🚀 How to Run
Clone the repository:
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install dependencies:
pip install -r requirements.txt
Update dataset path in notebook:
data = pd.read_csv("path/to/creditcard.csv")
Run the Jupyter Notebook:
jupyter notebook
📌 Key Insights
Fraud detection is highly affected by class imbalance
SMOTE + feature selection + tuned models significantly improves performance
Cost-sensitive learning is critical in real-world applications
Interpretability (SHAP) is essential for trust in financial systems
📈 Future Improvements
Deploy model using Flask / FastAPI
Real-time fraud detection pipeline
Try advanced deep learning models (LSTM, Autoencoders)
Integrate streaming data (Kafka)
👤 Author

Atharva Kocharekar
