# Assignment 5 — Loan Approval Classification

DS/ML Bootcamp | Lesson 5: Classification
Student: **Alisalad** (GitHub: [Ali-Salad](https://github.com/Ali-Salad))

## Overview

This project builds a binary classifier that predicts whether a loan application should be **Approved** or **Rejected**, based on applicant income, credit score, employment history, loan amount, collateral, and prior default history. It reproduces the in-class pipeline (Logistic Regression + Random Forest) and adds a third, self-researched algorithm: **K-Nearest Neighbors (KNN)**.

## Files

| File | Description |
|---|---|
| `loan_data_processing.ipynb` | Cleans the raw loan dataset: fixes currency formatting, normalizes Yes/No typos, imputes missing values, removes duplicates, caps outliers, label-encodes, engineers `DebtToIncome` and `IncomePerYearEmployed`, scales numeric features, and saves `clean_loan_dataset.csv`. |
| `loan_approval_prediction.ipynb` | Loads the cleaned dataset, splits it 80/20 (stratified), and trains Logistic Regression, Random Forest, and KNN classifiers. |
| `clean_loan_dataset.csv` | Cleaned, scaled dataset used for classification (100 rows, 9 columns). Note: currently uploaded/saved with a `.xls` extension but the file is plain CSV — should be renamed. |
| `paper.md` | Theory paper covering Classification concepts, all three algorithms, evaluation metrics, and results. |
| `reflection_paper.md` | Honest write-up of what worked, what didn't, and issues found in the notebooks as submitted. |

## Dataset

9 columns after preprocessing:

- `Income`, `CreditScore`, `EmploymentYears`, `LoanAmount` — scaled numeric features
- `HasCollateral`, `PreviousDefaults` — binary (0/1)
- `DebtToIncome`, `IncomePerYearEmployed` — engineered ratios
- `Approved` — target label (1 = Approved, 0 = Rejected); class split is 66% / 34%

## How to Run

```bash
pip install pandas numpy scikit-learn

# Step 1 — clean the raw data
jupyter nbconvert --to notebook --execute loan_data_processing.ipynb

# Step 2 — train and evaluate the classifiers
jupyter nbconvert --to notebook --execute loan_approval_prediction.ipynb
```

**Note:** as currently written, `loan_data_processing.ipynb`'s final save cell has a syntax error (`df.to_csv("clean_loan_dataset.csv" index=False)` is missing a comma), and `loan_approval_prediction.ipynb`'s evaluation cell is empty. Both need to be fixed before a clean top-to-bottom run will succeed — see `reflection_paper.md` for details.

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.700 | 0.733 | 0.846 | 0.786 |
| Random Forest | 0.650 | 0.714 | 0.769 | 0.741 |
| KNN (k=5) | 0.650 | 0.688 | 0.846 | 0.759 |

Full metric definitions, confusion matrices, and analysis are in `paper.md`; a critical look at the result quality (small sample size, why the Random Forest result contradicts the "usually more accurate" expectation) is in `reflection_paper.md`.
