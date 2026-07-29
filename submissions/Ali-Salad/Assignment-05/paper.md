# Assignment 5 — Classification: Loan Approval Prediction

**Author:** Alisalad (Ali-Salad)
**Track:** DS/ML Bootcamp — Lesson 5, Classification

---

## 1. Classification vs. Regression

**Q: What is Classification, and how is it different from Regression?**

Classification is a supervised learning task where the model predicts a **category** rather than a number. The target variable has a fixed set of labels — in this project, a loan application is either **Approved (1)** or **Rejected (0)**.

Regression, by contrast, predicts a **continuous value** — for example, predicting a house price or a car price (Assignment 4).

**Quick check:**

| Question | Answer |
|---|---|
| Predicting whether a loan is approved or rejected | Classification |
| Predicting the rent price of an apartment | Regression |
| Predicting whether a customer will churn | Classification |
| Predicting a patient's blood pressure reading | Regression |

Rule of thumb: **Regression = how much. Classification = which class.**

---

## 2. Algorithm 1 — Logistic Regression

Logistic Regression estimates the **probability** that an applicant belongs to the "Approved" class using a sigmoid function, then applies a 0.5 threshold to convert that probability into a hard label.

**Inputs used:** Income, CreditScore, EmploymentYears, LoanAmount, HasCollateral, PreviousDefaults, DebtToIncome, IncomePerYearEmployed
**Output:** Approved (1) or Rejected (0)

Example: an applicant with a strong credit score, low debt-to-income ratio, and no previous defaults should push the sigmoid output toward 1 (high probability of approval).

**Strengths:** fast, interpretable (coefficients show which features push toward approval/rejection), good baseline.
**Weaknesses:** assumes a roughly linear relationship between features and the log-odds of approval; struggles with complex, non-linear interactions between features.

---

## 3. Algorithm 2 — Random Forest

Random Forest builds many decision trees, each trained on a random subset of rows and features, and combines their votes into a final prediction (majority vote).

Example: if 100 trees each "look" at a slightly different slice of the applicant pool, and 68 of them vote "Approve," the forest's final prediction is Approved.

**Strengths:** captures non-linear patterns without manual feature engineering; generally more robust to noisy or interacting features than a single model.
**Weaknesses:** harder to interpret than Logistic Regression (no single coefficient explains a decision); more trees means more compute.

---

## 4. Algorithm 3 (Researched) — K-Nearest Neighbors (KNN)

KNN was the third algorithm required by the assignment. Unlike Logistic Regression and Random Forest, KNN does not "learn" a model during training — it just stores the training data. To classify a new applicant, it:

1. Measures the distance (Euclidean, by default) from the new applicant to every applicant in the training set.
2. Finds the **k** closest applicants (this project used **k = 5**).
3. Takes a majority vote of their labels — if 3 of the 5 nearest applicants were approved, the new applicant is predicted Approved.

Example: a new applicant with income, credit score, and loan amount similar to five previously-approved applicants is likely to be classified as Approved, purely by "resembling" them.

**Strengths:** simple, no assumptions about the shape of the data, can capture irregular decision boundaries.
**Weaknesses:** sensitive to feature scaling (an unscaled "Income" column in the tens of thousands would dominate distance calculations over a "CreditScore" column in the hundreds — this is why scaling with `RobustScaler` in preprocessing matters), slower to predict as the dataset grows, and sensitive to the choice of k.

### Algorithm Comparison

| Feature | Logistic Regression | Random Forest | KNN |
|---|---|---|---|
| Training phase | Learns coefficients | Builds trees | None (lazy learner) |
| Output | Class probability | Majority vote | Majority vote of neighbors |
| Interpretability | High | Lower | Low |
| Sensitive to feature scaling | Somewhat | No | Yes — strongly |
| Non-linear patterns | Limited | Yes | Yes, locally |

---

## 5. Evaluation Metrics

**Q: Why isn't Accuracy always enough?**

Accuracy can be misleading when classes are imbalanced. In this dataset, 66% of applicants were approved and 34% rejected. A model that blindly predicts "Approved" every time would score roughly 65–66% accuracy on this data without learning anything — which is close to what two of the three models below actually achieved.

| Metric | Question it answers | Example (this project's Logistic Regression result) |
|---|---|---|
| Accuracy | Of all predictions, how many were correct? | 14 of 20 test applicants correctly classified → 70% |
| Precision | Of applicants predicted "Approved," how many really should be? | 11 correctly approved out of 15 predicted approvals → 73.3% |
| Recall | Of applicants who really should be approved, how many did the model catch? | 11 of 13 truly-approvable applicants caught → 84.6% |
| F1-Score | Balance of Precision and Recall | 78.6% |
| Confusion Matrix | Full breakdown of TP, FP, FN, TN | See Section 6 |

In a loan context, **Precision** matters most when false approvals (lending to someone who defaults) are costly. **Recall** matters most when false rejections (turning away a good applicant) are costly. Neither metric alone tells the full story — this is why the Confusion Matrix and F1-Score exist.

---

## 6. Results on the Loan Dataset

Data: `clean_loan_dataset.csv` (100 rows after cleaning/dedup, 80/20 stratified train/test split, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.700 | 0.733 | 0.846 | 0.786 |
| Random Forest | 0.650 | 0.714 | 0.769 | 0.741 |
| KNN (k=5) | 0.650 | 0.688 | 0.846 | 0.759 |

**Confusion matrices** (rows = actual, columns = predicted; order is [Rejected, Approved]):

```
Logistic Regression        Random Forest              KNN (k=5)
              Pred                        Pred                        Pred
           Rej  App                    Rej  App                    Rej  App
Act Rej [   3    4  ]      Act Rej [   3    4  ]      Act Rej [   2    5  ]
Act App [   2   11  ]      Act App [   3   10  ]      Act App [   2   11  ]
```

**Single-row sanity check:** one held-out applicant with a genuinely strong profile (Approved = 1) was misclassified as "Rejected" by **all three models**, showing this applicant sat near the decision boundary shared by all three approaches.

**Best performer on this run:** Logistic Regression, by a narrow margin. See the reflection paper for why this result should be treated cautiously rather than as proof Logistic Regression is "better."

---

## 7. Summary

- Classification predicts categories (Approved/Rejected); regression predicts numbers.
- Logistic Regression gives interpretable probabilities and is a fast baseline.
- Random Forest handles non-linear relationships through many voting trees.
- KNN classifies by comparing a new applicant to its nearest neighbors in the training data — no training phase, but scaling-sensitive.
- Accuracy alone is not reliable on imbalanced data; Precision, Recall, F1-Score, and the Confusion Matrix together give the full picture of what a model gets right and where it fails.
