# 🧠 Credit Card Fraud Detection — End-to-End Machine Learning Project

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning.  
Due to the highly imbalanced nature of fraud data, the primary objective is to minimize false negatives (FN) and maximize recall for fraudulent transactions rather than relying on accuracy alone.

The project demonstrates a complete end-to-end machine learning workflow, from data exploration and model evaluation to deployment using a Flask API.

---

## 📊 Dataset
- Source: Kaggle – Credit Card Fraud Detection Dataset  
- Features:
  - V1–V28: PCA-transformed, anonymized features
  - Time, Amount
- Target Variable:
  - 0 → Legitimate transaction
  - 1 → Fraudulent transaction

The dataset is extremely imbalanced, reflecting real-world fraud detection scenarios.

---

## 🔄 Machine Learning Workflow
1. Exploratory Data Analysis (EDA)
2. Understanding and handling class imbalance
3. Model training and comparison
4. Evaluation using appropriate metrics
5. Final model selection
6. Model saving
7. Deployment using Flask API

---

## 🔍 Exploratory Data Analysis (EDA)
- Analyzed class distribution (fraud vs legitimate)
- Studied feature correlations
- Performed time-based transaction analysis
- Visualized transaction patterns

EDA insights guided the selection of evaluation metrics and sampling strategies.

---

## 🤖 Models Implemented
The following machine learning models were trained and evaluated:

- Logistic Regression (with undersampling on training data)
- Decision Tree
  - Class weighting
  - Gini and Entropy comparison
- Random Forest
  - Ensemble model with class weighting

Each model was evaluated on the original test dataset to simulate real-world performance.

---

## 📐 Evaluation Metrics
Because this is an imbalanced classification problem, the following metrics were prioritized:

- Recall (Fraud class)
- False Negatives (FN)
- F1-Score
- ROC-AUC
- Confusion Matrix

Accuracy was not used as the primary decision metric.

---

## ✅ Final Model Selection
Logistic Regression was selected as the final model because:
- It achieved the lowest number of false negatives
- It provided the highest recall for fraudulent transactions
- It aligned best with the business objective of fraud prevention

Although Decision Tree and Random Forest showed good performance, they missed more fraud cases compared to Logistic Regression.

---

## 🚀 Deployment
The final model was deployed using a Flask web application.

### API Details
- Endpoint: /predict
- Method: POST
- Input: Transaction feature values in JSON format
- Output:
  - 0 → Legitimate transaction
  - 1 → Fraudulent transaction

The trained model was saved as model.pkl and loaded into the Flask application for real-time predictions.
🗂️ Project Structure
<pre>
Credit_Card_Fraud_Detection/
│
├── data/                 # Dataset
├── notebooks/            # EDA and model experimentation
├── src/
│   └── models/           # Final training scripts
├── deployment/
│   ├── app.py            # Flask API
│   └── model.pkl         # Trained model
├── venv/                 # Virtual environment
└── README.md
</pre>
---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Flask
- Git & GitHub

---

## 🧠 Key Learnings
- Handling highly imbalanced datasets
- Selecting evaluation metrics aligned with business goals
- Comparing models using recall and false negatives
- Converting notebooks into production-ready scripts
- Deploying machine learning models using Flask APIs

---

## 🎯 Conclusion
This project demonstrates a complete, real-world machine learning pipeline for credit card fraud detection.  
By prioritizing recall and minimizing false negatives, the solution aligns technical decisions with real business impact and is suitable for practical deployment scenarios.


