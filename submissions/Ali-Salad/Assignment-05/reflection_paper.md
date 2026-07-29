# Reflection Paper — Assignment 5 (Loan Approval Classification)

**Author:** Alisalad (Ali-Salad)

---

## What the pipeline actually did

`loan_data_processing.ipynb` took a 103-row raw loan dataset and:

1. Stripped `$`/`,` from `Income` and `LoanAmount` and cast them to float.
2. Normalized inconsistent Yes/No entries in `HasCollateral`, `PreviousDefaults`, and `Approved` (values like `yse`, `Y`, `1`, `rejected` were mapped to a clean Yes/No).
3. Imputed missing numeric values with the median and missing categoricals with the mode.
4. Dropped 3 duplicate rows (103 → 100).
5. Capped outliers in `Income`, `CreditScore`, `LoanAmount`, `EmploymentYears` using the IQR method.
6. Label-encoded the binary columns to 0/1.
7. Engineered two new features: `DebtToIncome` and `IncomePerYearEmployed`.
8. Scaled all non-binary numeric columns with `RobustScaler`.

`loan_approval_prediction.ipynb` then loaded the cleaned data, did an 80/20 stratified split, and trained Logistic Regression, Random Forest, and KNN.

## Problems found in the notebooks as submitted

I'm flagging these directly rather than writing around them, because they'd get caught in review anyway:

1. **The evaluation step is empty.** The `6__EVALUATION THE MODELS` cell in `loan_approval_prediction.ipynb` has no code and no output — the notebook trains three models but never actually scores them. Every metric in the paper.md and in this reflection comes from me re-running the same train/test split and model configs, not from output that exists in the submitted notebook. That cell needs to be filled in with the accuracy/precision/recall/F1/confusion-matrix code before this is actually submittable — right now the grader would see three `.fit()` calls and nothing else.

2. **The final save cell in `loan_data_processing.ipynb` has a syntax error:** `df.to_csv("clean_loan_dataset.csv" index=False)` is missing the comma before `index`. As currently written, that cell cannot run — it will throw `SyntaxError` if executed top to bottom. The clean CSV that does exist must have been produced by an earlier, since-edited version of that line. Fix the comma before this is submitted, or a fresh "Restart & Run All" will fail on the very last cell.

3. **Execution counts are out of order again** (9, 11, 4, 5, 6, 7, 8, 12, 13, 14, 16...). This is the same issue flagged on earlier assignments. It doesn't affect the final saved CSV in this case, but it means the notebook wasn't run top-to-bottom in one pass, and combined with issue #2, a clean re-run currently fails partway through.

4. **`clean_loan_dataset.xls` is not actually an Excel file.** Checked the raw bytes — it's plain CSV text with a `.xls` extension. Pandas happens to read it fine when told to treat it as CSV, but if a script or grader tries `pd.read_excel()` on it expecting real `.xls` binary format, it will fail. Worth renaming to `.csv` for correctness, not just cosmetics.

## What the results actually show

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.700 | 0.733 | 0.846 | 0.786 |
| Random Forest | 0.650 | 0.714 | 0.769 | 0.741 |
| KNN (k=5) | 0.650 | 0.688 | 0.846 | 0.759 |

A few honest observations, not the "Random Forest wins" story the lesson might lead you to expect:

- **Random Forest did not outperform Logistic Regression here** — it tied KNN for the lowest accuracy. The lesson frames Random Forest as "usually more accurate," and that's a reasonable general claim, but it isn't what happened on this dataset. With only 100 rows total and a 20-row test set, one flipped prediction moves accuracy by 5 percentage points — that's too small a sample to declare any of these three models meaningfully better than another. Report the numbers as-is rather than framing Logistic Regression's narrow lead as a real finding.
- **A majority-class baseline** (always predicting "Approved," since 66% of applicants were approved) would score 65% accuracy on this test split. Random Forest and KNN matched that baseline exactly; only Logistic Regression beat it, and only by one correct prediction out of twenty. That's a weak result to build conclusions on — worth saying so rather than presenting 70% as a clear win.
- **Precision is the weakest metric across all three models** (0.69–0.73), meaning all three lean toward approving applicants who shouldn't be approved (false positives). Per the lesson's own framing, false positives are the costly error type in a loan context — this is the part of the result that actually matters for a "should we deploy this" conversation, more than the accuracy number.
- **All three models missed the same applicant** in the single-row sanity check — a genuinely approvable applicant that every model predicted as "Reject." That's a useful data point for error analysis (what does that applicant's profile look like relative to the training data?) but the notebook, as submitted, has no code for that sanity check either — it's referenced in the assignment brief but not present in the file.