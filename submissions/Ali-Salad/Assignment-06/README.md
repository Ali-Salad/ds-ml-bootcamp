# Assignment 6 — Wholesale Customer Segmentation

DS/ML Bootcamp | Lesson 6: Clustering
Author: **Alisalad** (GitHub: [Ali-Salad](https://github.com/Ali-Salad))

## Overview

This project segments **440 wholesale clients** of a food distributor into groups based on annual spend across six product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen). It reproduces the in-class K-Means pipeline and adds a second, self-researched algorithm — **Agglomerative (Hierarchical) Clustering** — for comparison.

## Files

| File | Description |
|---|---|
| `customer_segmentation.ipynb` | Loads the raw dataset, IQR-caps and scales the six spend columns, runs the Elbow Method (k=1–10), fits K-Means (k=5) and Agglomerative Clustering (k=5), evaluates both with Silhouette Score and Davies–Bouldin Index, and saves the labeled dataset. |
| `segmented_wholesale_customers.csv` | Output dataset: original columns plus `KMeans_Cluster` and `Agg_Cluster` labels (440 rows). Note: currently uploaded/saved with a `.xls` extension but the file is plain CSV — should be renamed. |
| `paper.md` | Part A theory: unsupervised learning, K-Means vs. Hierarchical vs. DBSCAN, clustering metrics, choosing k, segment interpretation, and a real-world case study. |
| `reflection_paper.md` | Part C reflection: what was implemented, segment interpretation with business actions, K-Means vs. Agglomerative comparison, and issues found in the notebook as submitted. |

## Dataset

Source: UCI Wholesale Customers dataset (440 B2B clients).

- `Channel` — 1 = Horeca, 2 = Retail (excluded from clustering, kept for reference)
- `Region` — 1 = Lisbon, 2 = Oporto, 3 = Other (excluded from clustering, kept for reference)
- `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen` — annual spend per category (IQR-capped, then scaled with `StandardScaler` before clustering)
- `KMeans_Cluster`, `Agg_Cluster` — cluster labels added by the notebook

## How to Run

```bash
pip install pandas numpy scikit-learn

jupyter nbconvert --to notebook --execute customer_segmentation.ipynb
```

**Note:** the notebook's final save cell currently shows a stale error output (`Cannot save file into a non-existent directory: 'dataset'`) left over from an earlier version of that line. The code as currently written (`df.to_csv("segmented_wholesale_customers.csv", index=False)`) runs fine — do a **Restart & Run All** before submitting so the displayed output matches the current code. See `reflection_paper.md` for details.

## Results Summary

| Method | Silhouette Score | Davies–Bouldin Index |
|---|---|---|
| K-Means (k=5) | 0.283 | 1.270 |
| Agglomerative (Ward, k=5) | 0.218 | 1.325 |

K-Means produced better-separated clusters on this dataset. Cluster sizes range from 25 to 191 clients; cross-tabulating against the (unused-in-clustering) `Channel` column shows the clusters do largely track real Horeca vs. Retail purchasing patterns — see `reflection_paper.md` for the per-cluster breakdown and suggested business actions.

**Caveat:** a Silhouette Score of 0.28 indicates real but modest cluster separation, not five sharply distinct customer types — some cluster boundaries are soft, which is expected given how much overlap exists in real B2B purchasing behavior.
