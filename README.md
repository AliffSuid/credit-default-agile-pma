
# Credit Default Risk Decision Support using Agile Data Science

This project applies Agile Data Science to build a credit default risk decision support prototype using the Default of Credit Card Clients dataset.

## Project Objective

The objective of this project is to identify customers who are more likely to default in the next month so that risk or collection officers can prioritise high-risk accounts for early review.

## Dataset

Dataset: Default of Credit Card Clients  
Source: UCI Machine Learning Repository  
Records: 30,000  
Target variable: DEFAULT_NEXT_MONTH  
Prediction type: Binary classification

## Agile Sprint Summary

### Sprint 1: Data Understanding and Data Quality
- Performed dataset preview and summary
- Conducted EDA
- Checked missing values, duplicates, invalid categories, and outliers
- Produced cleaned dataset

### Sprint 2: Baseline Model
- Developed Logistic Regression baseline model
- Evaluated accuracy, precision, recall, F1-score, and ROC-AUC

### Sprint 3: Improved Model
- Developed Random Forest model
- Improved recall and F1-score for default customer detection

### Sprint 4: Dashboard and Monitoring
- Planned Streamlit dashboard
- Planned monitoring metrics and future backlog improvements

## Automation

The project includes an automated validation script:
- validate_data.py

The script checks missing values, duplicates, target column, valid categories, and infinite values.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- GitHub
- GitHub Actions
- Streamlit
