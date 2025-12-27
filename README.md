# Customer Churn Modeling

## Overview
This project focuses on predicting customer churn in the banking sector using supervised machine learning techniques. The objective is to identify customers at high risk of churn by analyzing demographic, financial, and behavioral features, enabling data-driven retention strategies.

## Dataset
The dataset contains information for 10,000 banking customers, including:
- Demographic attributes (age, gender, geography)
- Financial attributes (account balance, credit card ownership)
- Behavioral attributes (number of products, activity status)
- Churn indicator (target variable)

## Methodology
- Performed exploratory data analysis (EDA) to understand churn patterns across customer segments
- Preprocessed features and addressed class imbalance using SMOTE
- Trained and evaluated multiple supervised learning models, including:
  - Logistic Regression
  - Random Forest (with and without hyperparameter tuning)
  - Support Vector Machine (SVM)
  - Artificial Neural Network (ANN)
  - XGBoost
- Compared model performance using precision, recall, and ROC-AUC
- Analyzed feature-driven churn patterns to support business interpretation

## Results
- Class imbalance handling significantly improved recall and overall model performance
- Model comparisons highlighted trade-offs between interpretability and predictive power
- Analysis revealed high-risk churn segments based on product ownership, balance thresholds, age, and geography

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Key Takeaways
This project demonstrates an end-to-end machine learning workflow for churn prediction, emphasizing data preprocessing, imbalanced classification, model evaluation, and translation of analytical results into actionable business insights.
