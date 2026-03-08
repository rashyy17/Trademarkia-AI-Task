## Cluster Summary — 20 Newsgroups (generated clusters)

This file summarizes the fuzzy clusters discovered in the 20 Newsgroups corpus during preprocessing and clustering (PCA + GaussianMixture). It is intended for inclusion in the submission and explains cluster semantics, exemplar documents, and boundary/uncertain cases.

High-level
- Number of clusters chosen by BIC: 8
- Method: PCA (pca_dim=50) followed by GaussianMixture; clustering returns soft distributions (predict_proba) for each document.

Top terms per cluster (human-readable labels)
- Cluster 0 (cars / sales): car, new, price, sale, like, good
- Cluster 1 (X/windows / programs): windows, file, thanks, program, window, use
- Cluster 2 (politics / government / Israel / guns): people, government, israel, don, gun, just
- Cluster 3 (religion / Christianity): god, jesus, people, bible, believe, don
- Cluster 4 (sports / games): game, team, games, year, hockey, baseball
- Cluster 5 (encryption / policy): key, space, clipper, encryption, government
- Cluster 6 (food / casual topics / lifestyle): just, like, don, car, bike, know
- Cluster 7 (hardware / drives / mac / scsi): drive, card, scsi, monitor, mac, windows

Representative exemplars (top snippet per cluster)
- Cluster 0 (cars / sale)
  - doc 16986: "For sale - Mazda 323 1986 Mazda 323 ... 75,000 miles ... Interior in very good condition..."

- Cluster 1 (X/windows / programs)
  - doc 11571: "...a click of MB3 (right) automatically kills all clients - oh my :-( ..."

- Cluster 2 (politics / government / Israel / guns)
  - doc 7294: "Is this a figment of your imagination? ... Mitteilungsblatt, Berlin, December 1939 ... Armenian-Nazi collaboration..."

- Cluster 3 (religion)
  - doc 3029: "Before you finalize your file in the FAQs ... It seems one or the other end of the rating scale should be identified with 'homosexual'..."

- Cluster 4 (sports)
  - doc 0: "I am sure some bashers of Pens fans are pretty confused about the lack of any kind of posts about the recent Pens massacre of the Devils..."

- Cluster 5 (crypto / policy)
  - doc 14410: "...Lets write a DOCUMENT which includes all the reasons we oppose Clipper..."

- Cluster 6 (food / casual)
  - doc 9166: "MSG is mono sodium glutamate... My experience of MSG effects..."

- Cluster 7 (hardware / mac / drives)
  - doc 3909: "It is very possible to connect another internal hard disk in any macintosh if you can find the space to put it..."

Boundary / uncertain documents
- The clustering is fuzzy; some documents have high entropy across cluster assignments. These are interesting boundary cases.
- Example uncertain docs (entropy and top three probabilities):
  - doc 13, entropy=1.1980, top_probs=[0.356, 0.330, 0.283] — discussion about Kirlian imaging (ambiguous between tech/science/other)
  - doc 3215, entropy=1.1884, top_probs=[0.427, 0.355, 0.161] — discussion of grounding and shielding (electronics vs general engineering)
  - doc 10690, entropy=1.1790, top_probs=[0.486, 0.291, 0.157] — gate minimization / hardware timing (mix of hardware and algorithms)

Interpretation notes
- Clusters are semantically meaningful: top TF-IDF terms and exemplar snippets align with intuitive topical categories (cars, software, politics, religion, sports, crypto/policy, lifestyle, hardware).
- Boundary documents (high entropy) often combine topics or are short and ambiguous; they illustrate why soft assignments are necessary for downstream tasks like semantic caching.

How clusters are used in cache
- The semantic cache partitions entries by dominant cluster and only compares query embeddings against cached entries in the top-K clusters for the query (default top_k=3). This reduces lookup cost as the cache grows and leverages semantic structure to improve hit efficiency.

Files produced
- `data/cluster_report.txt` — full machine-generated report with top 20 TF-IDF terms per cluster, top exemplars, and a longer list of uncertain documents.

Suggested submission excerpts
- Include the `Top terms per cluster` table and 1–2 exemplar snippets per cluster (we included the top example above for each cluster). Add a short paragraph about the boundary cases (we provided three examples). These demonstrate the semantic coherence of clusters and the presence of genuine ambiguities.

Next steps (optional)
- Add a small t-SNE or UMAP visualization of PCA-reduced embeddings colored by dominant cluster to include in the submission PDF or README (I can generate this and a PNG).
- Include 5–10 hand-checked boundary documents with short annotations explaining why they sit between clusters.

---
Generated from `scripts/cluster_report.py` output. For more detail and the full list of uncertain documents, see `data/cluster_report.txt`.
