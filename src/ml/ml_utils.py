import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
import streamlit as st
from kneed import KneeLocator
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from umap import UMAP

from arxiv_searcher import Paper
from .preprocessing import (
    get_top_k_words,
    preprocess_texts,
    get_top_k_words_ctfidf,
)

MIN_PAPERS_FOR_CLUSTERING = 15
MAX_NUMBER_OF_CLUSTERS = 15
_DEFAULT_METRICS = {"SIL": 0, "DBI": 0, "CHI": 0}


@st.cache_resource(show_spinner=False)
def get_optimal_k(papers: List[Paper]) -> int:
    """
    Find the optimal number of clusters for K-Means clustering
    using the elbow method on Inertia.
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return 1

    try:
        X_normalized = preprocess_texts(papers)
        K_range = range(2, MAX_NUMBER_OF_CLUSTERS)
        inertias = []
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, init="k-means++", n_init=1, max_iter=300)
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


#@st.cache_resource(show_spinner=False)
def get_papers_kmeans(papers: List[Paper], n_clusters: int = 5):
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    if n_clusters > MAX_NUMBER_OF_CLUSTERS:
        n_clusters = MAX_NUMBER_OF_CLUSTERS

    try:
        X_normalized = preprocess_texts(papers)
        model = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=1, max_iter=300)
        labels = model.fit_predict(X_normalized)

        # TODO remove this for deployment
        # plot_clusters_umap(is_hdbscan=False, X=X_normalized, labels=labels)

        clusters: Dict[int, list] = {i: [] for i in range(n_clusters)}
        
        for paper_idx, paper in enumerate(papers):
            clusters[int(labels[paper_idx])].append(paper)
            
        # Drop clusters that ended up empty
        clusters = {k: v for k, v in clusters.items() if v}

        top_clusters_words = get_top_k_words_ctfidf(
            papers, labels, top_k=4
        )

        if len(np.unique(labels)) >= 2:
            metrics = {
                "SIL": silhouette_score(X_normalized, labels, metric="cosine"),
                "DBI": davies_bouldin_score(X_normalized, labels),
                "CHI": calinski_harabasz_score(X_normalized, labels),
            }
        else:
            metrics = _DEFAULT_METRICS

        print(f"K-Means Clustering: {len(clusters)} clusters")
        print(f"Metrics: {metrics}")
        print(top_clusters_words)
        print("=" * 50 + "\n")

        return dict(sorted(clusters.items())), top_clusters_words, metrics

    except Exception as e:
        logging.error(f"K-Means Clustering error: {e}")
        return {0: papers}, {}, _DEFAULT_METRICS


@st.cache_resource(show_spinner=False)
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
        X_normalized = preprocess_texts(papers)
        model = HDBSCAN(
            min_samples=1,
            cluster_selection_epsilon=0.2,
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

        # Get the real (non-noise) labels
        real_labels = sorted(l for l in np.unique(labels) if l != -1)

        # Build clusters dict using native HDBSCAN labels
        clusters: Dict[int, list] = {}
        for paper_idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(papers[paper_idx])

        # Drop any cluster that ended up empty
        clusters = {k: v for k, v in clusters.items() if v}

        if model.centroids_ is None or len(model.centroids_) == 0:
            return {0: papers}, {}, _DEFAULT_METRICS

        # top_words: get_top_k_words_ctfidf returns a list indexed 0..K-1,
        # one entry per real cluster in sorted label order.
        raw_top_words = get_top_k_words_ctfidf(
            papers, real_labels, top_k=3
        )

        # Map each real cluster label directly to its top words
        top_words: Dict[int, list] = {label: raw_top_words[i] for i, label in enumerate(real_labels)}
        # Noise cluster has no top words
        top_words[-1] = []

        # Metrics (evaluated on non-noise points only)
        non_noise_mask = labels != -1
        y_eval = labels[non_noise_mask]
        X_eval = X_normalized[non_noise_mask]

        if len(np.unique(y_eval)) >= 2:
            metrics = {
                "SIL": silhouette_score(X_eval, y_eval, metric="cosine"),
                "DBI": davies_bouldin_score(X_eval, y_eval),
                "CHI": calinski_harabasz_score(X_eval, y_eval),
            }
        else:
            metrics = _DEFAULT_METRICS

        noise_ratio = float(np.mean(labels == -1))

        # TODO remove this for deployment
        plot_clusters_umap(is_hdbscan=True, X=X_normalized, labels=labels)

        print(f"HDBSCAN Clustering: {len(clusters)} clusters (including noise)")
        print(f"Metrics: {metrics}")
        print(f"Noise Ratio: {noise_ratio}")

        return dict(sorted(clusters.items())), top_words, metrics

    except Exception as e:
        logging.error(f"HDBSCAN Clustering error: {e}")
        return {0: papers}, {}, _DEFAULT_METRICS


def plot_clusters_umap(is_hdbscan, X, labels, save_path="clusters_umap.png"):
    K = len(np.unique(labels))
    n_neighbors = max(2, min(X.shape[0] - 1, X.shape[0] // K))

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=42,
    )
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for cluster in np.unique(labels):
        mask = labels == cluster
        legend_label = "Noise" if (cluster == -1 and is_hdbscan) else f"Cluster {cluster + 1}"
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=12,
            alpha=0.75,
            label=legend_label
        )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("Clusters (UMAP projection)")
    plt.legend(title="Clusters", markerscale=2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()