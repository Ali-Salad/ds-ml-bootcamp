# Reflection Paper — Assignment 6 (Wholesale Customer Segmentation)

**Author:** Alisalad (Ali-Salad)

---

## 1. What I Implemented

I segmented 440 wholesale clients using their annual spend on six product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen). After IQR-capping outliers and scaling with `StandardScaler`, I ran K-Means (k=5, matching the lesson) and, as the required second method, **Agglomerative (Hierarchical) Clustering** with Ward linkage, also cut into 5 clusters. `Channel` and `Region` were kept in the dataframe for reference but excluded from the clustering features, as required.

## 2. Segment Interpretation

Using the K-Means cluster centers (converted back to original spend units) and cross-tabulating clusters against the known `Channel` column:

- **Cluster 3 (88 clients, 82 Horeca / 6 Retail):** Very high Fresh (~22,347) and Frozen (~5,820) spend, low Grocery and Detergents_Paper. This is the classic **Horeca profile** — hotels and restaurants buying perishables to prepare food, with little need for shelf-stable groceries or cleaning supplies. **Business action:** prioritize fast, frequent fresh/frozen delivery routes and freshness-based promotions for this group rather than bulk grocery discounts, which this segment barely buys.

- **Cluster 4 (60 clients, 2 Horeca / 58 Retail):** Low Fresh (~4,917), high Grocery (~18,350) and Detergents_Paper (~7,780). This is the **Retail profile** — shops reselling packaged goods and cleaning supplies. **Business action:** bundle Grocery and Detergents_Paper into volume-discount packages, since that's essentially the entire wallet for this segment.

- **Cluster 2 (25 clients, mixed Channel):** The smallest cluster but the highest spend across *every* category (Fresh ~17,462, Milk ~13,806, Grocery ~17,524). These are the distributor's largest accounts regardless of Horeca/Retail type. **Business action:** assign dedicated account managers — losing even one of these 25 clients has an outsized revenue impact compared to losing a client from the 191-client Cluster 1.

## 3. Understanding K-Means

In my own words: K-Means tries to find *k* "center points" (centroids) that best summarize the data. It starts by placing centroids (randomly, or via smarter initialization), then repeats two steps — assign every client to whichever centroid is closest, then move each centroid to the average position of the clients now assigned to it — until the centroids stop moving. The result is *k* groups where clients inside a group are, on average, closer to their own group's center than to any other group's center. It's fast and easy to explain, but it can only really "see" round, evenly-sized clusters, and a bad choice of *k* or an unlucky starting position can lead to a poor grouping.

## 4. My Second Algorithm — Agglomerative Clustering

I chose **Agglomerative Clustering with Ward linkage** because it doesn't force an assumption of round, evenly-sized clusters the way K-Means does, and because it's the natural "next algorithm" to compare against K-Means on the same numeric spend data without needing density parameters (which DBSCAN would require, and which are harder to tune well on a first pass).

What I learned researching it: it builds clusters bottom-up, starting with every client as its own cluster and merging the two closest clusters repeatedly until only 5 remain (in this case). **Advantage:** no need to pre-specify *k* the way K-Means does — you can cut the merge tree (dendrogram) at whatever number of groups makes sense afterward. **Limitation:** merges are permanent and greedy — an early merge that turns out to be a poor grouping decision can't be undone later in the process, unlike K-Means where every point can be reassigned each iteration.

**Silhouette comparison:** K-Means scored **0.283**, Agglomerative scored **0.218**. K-Means produced the better-separated clusters on this dataset.

## 5. Findings and Recommendation

I'd recommend **K-Means** for this specific segmentation task — it scored a higher Silhouette (0.283 vs. 0.218) and a lower (better) Davies–Bouldin index (1.270 vs. Agglomerative's 1.325, computed the same way), and its cluster sizes (25–191 clients) are more evenly usable for a sales team than Agglomerative's output. That said, neither score is particularly strong on its own — a Silhouette of 0.28 indicates real but modest separation, not sharply distinct customer types, which tracks with how wholesale spending actually works: a hotel and a large restaurant can have genuinely overlapping purchase patterns. I would not oversell the "5 distinct customer types" framing to the business without noting that the boundaries between some clusters are soft.

## Issues Found in the Notebook

Flagging these directly rather than glossing over them:

- **Execution counts in `customer_segmentation.ipynb` are sequential (6 through 15)** — unlike the loan assignment, this notebook was actually run top-to-bottom in order, which is the right habit and worth keeping up.
- **The final save cell has a stale error output that no longer matches its own code.** The cell's *source* now reads `df.to_csv("segmented_wholesale_customers.csv", index=False)`, but the *displayed traceback* is from an older version of the cell that tried to save to `"dataset/segmented_wholesale_customers.csv"` and failed because that folder doesn't exist. In other words, the cell was edited after it last ran, and the notebook was never re-executed to refresh the output — so the file currently shows an error message next to code that would actually succeed if run. This is the same "run top-to-bottom before submitting" issue flagged on earlier assignments; it happened to work out here because the uploaded `segmented_wholesale_customers.csv` matches what the current code would produce, but that's the one save cell in the whole notebook and it's the one showing a red error box. Worth a Restart & Run All before submission so the visible output matches the visible code.
- **The elbow in the SSE curve isn't as clean as the lesson's toy example.** The actual SSE values are: k=1: 2640.00, k=2: 1728.19, k=3: 1363.45, k=4: 1202.41, **k=5: 1070.15**, k=6: 964.76, k=7: 921.14, k=8: 776.63, k=9: 726.88, k=10: 707.41. The percentage drop from k=7→8 (15.7%) is actually larger than the drop from k=4→5 (11.0%) or k=6→7 (4.5%) — the curve isn't smoothly diminishing, so picking k=5 "because of the elbow" is a weaker justification than it looks on a textbook chart. k=5 is a reasonable choice here because it matches the assignment's instructions and gives business-usable segment sizes, not because the SSE curve points there unambiguously on its own. Worth saying so rather than claiming the elbow method "confirmed" k=5.
