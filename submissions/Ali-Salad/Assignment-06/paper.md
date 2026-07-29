# Assignment 6 — Clustering: Theory

**Author:** Alisalad (Ali-Salad)
**Track:** DS/ML Bootcamp — Lesson 6, Clustering

---

## 1. Introduction to Unsupervised Learning and Clustering

**Unsupervised learning** is a branch of Machine Learning where the model is given data with no target labels and has to find structure in it on its own — there is no "correct answer" column to learn from, only the input features.

This is fundamentally different from the regression and classification work in Lessons 4 and 5. In those lessons, every row had a known outcome (a car price, a loan approval decision), and the model's job was to learn the mapping from features to that known outcome. In clustering, there is no outcome column at all. The algorithm groups rows based on how similar they are to each other, and it is up to the analyst — not the algorithm — to interpret what each resulting group represents.

**Example of clustering:** grouping wholesale clients by what they buy (this project) — the distributor has no pre-existing "customer type" label, only six spending columns, and K-Means discovers the groupings.

**Example of supervised learning:** predicting whether a loan application should be approved or rejected (Assignment 5) — every historical row already has an Approved/Rejected label to learn from.

---

## 2. Clustering Algorithms

### K-Means

**How it works:** K-Means partitions the data into a fixed number of clusters, *k*, chosen in advance. It places *k* centroids, assigns every point to its nearest centroid, recalculates each centroid as the mean of its assigned points, and repeats the assign-and-update cycle until the centroids stop moving.

**Real-world use case:** Segmenting wholesale or retail customers by purchasing behavior so a business can target each group with a different strategy (this project).

**Advantages:** Fast and scales well to medium/large datasets; simple to explain to non-technical stakeholders (each cluster has an interpretable "average customer" — the centroid).
**Limitations:** *k* must be chosen ahead of time; assumes roughly round (spherical), similarly-sized clusters; sensitive to outliers and to feature scale, which is why scaling is a required preprocessing step.

### Hierarchical (Agglomerative) Clustering

**How it works:** Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest clusters until only one cluster remains, building a tree of merges (a **dendrogram**). Cutting the tree at a chosen height produces a specific number of clusters. This project used **Ward linkage**, which merges the two clusters whose combination increases within-cluster variance the least.

**Real-world use case:** Grouping genes with similar expression patterns in biology, or organizing documents/products into a taxonomy where the natural nesting of groups (sub-groups within groups) is itself useful information.

**Advantages:** Does not require choosing *k* in advance — the dendrogram lets you inspect groupings at every level; can capture nested/hierarchical structure that K-Means cannot.
**Limitations:** Computationally expensive on large datasets (distance calculations grow roughly quadratically with the number of points); merges are greedy and permanent — an early bad merge can't be undone later in the process.

### DBSCAN

**How it works:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are packed closely (within a distance `eps`, with at least `min_samples` neighbors), and labels points that don't belong to any dense region as **noise/outliers** rather than forcing them into a cluster.

**Real-world use case:** Detecting anomalies or fraud in transaction data, where the "clusters" are normal-behavior groups and the unclustered points are exactly the fraudulent or anomalous cases worth investigating.

**Advantages:** Does not require specifying the number of clusters; can find irregularly-shaped clusters that K-Means (which assumes round clusters) would split incorrectly; naturally flags outliers instead of forcing every point into a group.
**Limitations:** Sensitive to the `eps` and `min_samples` parameters, which can be hard to tune; struggles when clusters have very different densities, since a single `eps` value may fit some clusters well and not others.

### Comparison Summary

| | K-Means | Hierarchical (Agglomerative) | DBSCAN |
|---|---|---|---|
| Must choose k in advance | Yes | No (cut dendrogram after) | No (finds clusters by density) |
| Handles non-round clusters | No | Somewhat | Yes |
| Handles outliers | No — every point is assigned | No — every point is assigned | Yes — labels them as noise |
| Scales to large data | Well | Poorly (quadratic distance matrix) | Moderately |

---

## 3. Clustering Metrics

Since there are no true labels in clustering, we cannot compute accuracy or F1-score. Instead, we judge cluster quality using the geometry of the clusters themselves.

- **Elbow Method (SSE):** Sum of squared distances from each point to its assigned centroid, computed for a range of *k* values. As *k* increases, SSE always decreases (more clusters means points sit closer to their centroid), but the rate of improvement slows down. The "elbow" — where adding another cluster stops giving a meaningful reduction in SSE — is used as a rough guide for choosing *k*. It is a visual/manual judgment call, not an exact formula.

- **Silhouette Score:** For each point, compares its average distance to points in its own cluster against its average distance to points in the nearest other cluster. Ranges from -1 to +1; scores near +1 mean the point sits comfortably inside a well-separated cluster, scores near 0 mean it sits on the border between two clusters, and negative scores mean the point is probably in the wrong cluster. Averaged across all points, it gives a single overall cluster-quality number.

- **Davies–Bouldin Index:** Measures the average "similarity" between each cluster and its most similar other cluster, where similarity combines how spread out the clusters are with how far apart their centers are. Unlike Silhouette, **lower is better** — a low score means clusters are compact and well-separated from their nearest neighbor cluster.

### Comparison Table

| Metric | What It Measures | Good Value Looks Like |
|---|---|---|
| Elbow (SSE) | Compactness of clusters as k increases | A visible bend where SSE stops dropping quickly |
| Silhouette Score | Cohesion within a cluster vs. separation from others | Close to +1 |
| Davies–Bouldin Index | Average similarity between each cluster and its closest neighbor | Close to 0 (lower is better) |

---

## 4. Choosing k and Interpreting Segments

**Choosing k:** In practice, *k* is chosen using a combination of the Elbow Method (to see roughly where SSE improvement slows), the Silhouette Score (to confirm that the chosen *k* actually produces well-separated groups, not just lower SSE), and business judgment (a distributor's sales team can realistically act on 4–6 customer types, not 20). None of these methods gives one "correct" answer on their own — they're used together as evidence.

**Interpreting spend patterns:** In the wholesale distributor project, a cluster with high **Fresh + Milk** spend and low Grocery/Detergents_Paper typically represents a **Horeca client** (hotel, restaurant, or café) — businesses that buy perishables to prepare and serve food, and don't need bulk packaged groceries or cleaning supplies for resale. A cluster with high **Grocery + Detergents_Paper** spend and comparatively low Fresh spend typically represents a **Retail client** (a shop reselling packaged goods) — they stock shelf-stable products and cleaning supplies, not perishables that spoil quickly on a store shelf.

**Why Channel and Region are excluded from clustering:** `Channel` (Horeca vs. Retail) and `Region` are **known, pre-existing labels**, not spending behavior. Including them as clustering features would let the algorithm "cheat" by grouping clients using a label we already have, instead of discovering new structure from actual purchasing patterns. Keeping them out of the feature set — but still attached to the dataframe — lets us afterward check whether the *discovered* clusters line up with the *known* Channel/Region categories, which is a useful way to validate whether the clustering found something meaningful.

---

## 5. Real-World Case Study

**Study:** Chen, D. et al., *"Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining,"* published in the *Journal of Database Marketing & Customer Strategy Management* (2012).

**Goal:** the study's purpose was to help a small UK-based online retailer of gift items better understand its customer base so it could run more effective, customer-centric marketing (Chen, Sain & Guo, 2012).

**Data used:** transaction-level records from the retailer, covering purchase dates, quantities, and prices — enough to compute a Recency, Frequency, and Monetary (RFM) score for each customer. This RFM approach on online retail transaction data has since become one of the most widely reused datasets and methodologies in customer segmentation research.

**Method:** customers were segmented into meaningful groups using K-Means clustering, combined with decision tree induction to characterize each segment's defining traits (Chen et al., 2012). This mirrors the wholesale project's approach: reduce raw transaction/spend data to a small set of numeric features, scale them, and let K-Means find groups.

**Key results/insights:** the characteristics of customers within each resulting segment were clearly identified, and the study concluded with a set of concrete recommendations for customer-centric marketing tailored to each group (Chen et al., 2012). The broader lesson — echoed in follow-up research that continues to use K-Means for RFM-based segmentation — is that grouping customers by a handful of behavioral numbers (recency, frequency, monetary value; or, in the wholesale case, spend per product category) reliably surfaces actionable groups like "high-value regulars" versus "price-sensitive occasional buyers," even without any demographic labels.

**Reference:** Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*(3), 197–208. https://link.springer.com/article/10.1057/dbm.2012.17

---

## 6. Summary

- Unsupervised learning finds structure without labeled outcomes; clustering specifically groups similar rows together.
- K-Means is fast and interpretable but requires choosing *k* upfront and assumes round clusters; Hierarchical clustering avoids pre-choosing *k* but doesn't scale well; DBSCAN handles irregular shapes and outliers but is sensitive to its distance parameters.
- Elbow, Silhouette, and Davies–Bouldin are used together — no single metric proves a clustering is "correct," since there are no ground-truth labels to check against.
- In the wholesale project, cluster interpretation should be grounded in spend patterns (Fresh/Milk vs. Grocery/Detergents_Paper), and Channel/Region are held out of the features specifically so they can be used afterward to sanity-check the discovered groups.
