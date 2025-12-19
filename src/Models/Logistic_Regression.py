"""
Final Logistic Regression Model
Credit Card Fraud Detection
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler


def train_and_save_model():
    # 1. Load dataset
    df = pd.read_csv("creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2. Train-test split (keep imbalance in test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Undersampling (ONLY on training data)
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    print("After undersampling:", y_train_resampled.value_counts())

    # 4. Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # 5. Predictions on original test data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Evaluation
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # 7. Save model
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved as model.pkl")


if __name__ == "__main__":
    train_and_save_model()
