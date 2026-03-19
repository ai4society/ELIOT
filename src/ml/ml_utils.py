import logging
from typing import Dict, List, Tuple

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from umap import UMAP

from arxiv_searcher import Paper
from utils import timeit

from .preprocessing import (
    embed_and_reduce,
    get_cluster_keywords,
    preprocess_texts,
)

MIN_PAPERS_FOR_CLUSTERING = 15
MAX_NUMBER_OF_CLUSTERS = 15
_DEFAULT_METRICS = {"SIL": 0, "DBI": 0, "CHI": 0}


@timeit
def get_optimal_k(papers: List[Paper]) -> int:
    """
    Find the optimal number of clusters using the elbow method
    on the K-means inertia curve.
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return 1

    logging.info("Calculating optimal number of clusters...")
    try:
        X_cleaned = preprocess_texts(papers)
        X_normalized = embed_and_reduce(X_cleaned)
        K_range = range(2, MAX_NUMBER_OF_CLUSTERS)
        inertias = []
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, init="k-means++", max_iter=300)
            kmeans_temp.fit(X_normalized)
            score = kmeans_temp.inertia_
            inertias.append(score)

        knee_locator = KneeLocator(list(K_range), inertias, curve="convex", direction="decreasing")
        # If no knee is found, return 2 as default
        n_clusters = int(knee_locator.knee) if knee_locator.knee else 2
        return n_clusters
    except Exception as e:
        logging.error(f"Optimal K error: {e}")
        return 1


@timeit
def get_papers_clusters_agglomerative(papers: List[Paper], n_clusters: int = 5):
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    if n_clusters > MAX_NUMBER_OF_CLUSTERS:
        n_clusters = MAX_NUMBER_OF_CLUSTERS

    try:
        X_cleaned = preprocess_texts(papers)
        X_normalized = embed_and_reduce(X_cleaned)
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = model.fit_predict(X_normalized)
        
        top_keywords = get_cluster_keywords(
            X_cleaned, labels, top_k=4
        )

        if len(np.unique(labels)) >= 2:
            metrics = {
                "SIL": float(silhouette_score(X_normalized, labels, metric="cosine")),
                "DBI": float(davies_bouldin_score(X_normalized, labels)),
                "CHI": float(calinski_harabasz_score(X_normalized, labels)),
            }
        else:
            metrics = _DEFAULT_METRICS

        clusters: Dict[int, list] = {i: [] for i in range(n_clusters)}
        
        for paper_idx, paper in enumerate(papers):
            clusters[int(labels[paper_idx])].append(paper)

        return dict(sorted(clusters.items())), top_keywords, metrics
    except Exception as e:
        logging.error(f"Agglomerative Clustering error: {e}")
        return {0: papers}, {}, _DEFAULT_METRICS


@timeit
def get_paper_clusters_hdbscan(papers: List[Paper]):
    """
    Performs HDBSCAN clustering on a list of papers.
    Returns:
        Tuple (clusters_dict, top_words_dict, metrics)
        - clusters: {cluster_id: [list_of_papers]}
        - top_words: {cluster_id: [top_words]}
        - metrics: Dictionary of clustering quality metrics
    """

    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    try:
        X_cleaned = preprocess_texts(papers)
        X_normalized = embed_and_reduce(X_cleaned)
        model = HDBSCAN(
            metric="euclidean",
            store_centers="centroid",
        )
        labels = model.fit_predict(X_normalized)

        # ------------------------------------------------------------------ #
        # Cluster dictionary construction
        #
        # HDBSCAN natively labels noise as -1 and real clusters as 0, 1, 2...
        # The UI shifts all keys by +1, so real clusters become 1, 2, 3...
        # and noise (-1) becomes 0.
        # ------------------------------------------------------------------ #

        # Build clusters dict using native HDBSCAN labels
        clusters: Dict[int, list] = {}
        for paper_idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(papers[paper_idx])

        # Drop any cluster that ended up empty
        clusters = {k: v for k, v in clusters.items() if v}

        if model.centroids_ is None or len(model.centroids_) == 0:
            logging.info("No centroids found (Predicted labels are all noise), returning single cluster")
            return {0: papers}, {}, _DEFAULT_METRICS

        # Extract keywords only for real (non-noise) points.
        top_words: Dict[int, list] = get_cluster_keywords(X_cleaned, labels, top_k=3)
        top_words[-1] = []  # noise cluster has no representative keywords

        # Metrics (evaluated on non-noise points only)
        non_noise_mask = labels != -1
        y_eval = labels[non_noise_mask]
        X_eval = X_normalized[non_noise_mask]

        if len(np.unique(y_eval)) >= 2:
            metrics = {
                "SIL": float(silhouette_score(X_eval, y_eval, metric="cosine")),
                "DBI": float(davies_bouldin_score(X_eval, y_eval)),
                "CHI": float(calinski_harabasz_score(X_eval, y_eval)),
            }
        else:
            metrics = _DEFAULT_METRICS

        return dict(sorted(clusters.items())), top_words, metrics

    except Exception as e:
        logging.error(f"HDBSCAN Clustering error: {e}")
        return {0: papers}, {}, _DEFAULT_METRICS
