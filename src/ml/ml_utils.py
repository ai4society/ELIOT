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
    get_top_k_words_from_svd_centroids,
    preprocess_texts,
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


@st.cache_resource(show_spinner=False)
def get_papers_kmeans(papers: List[Paper], n_clusters: int = 5):
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    if n_clusters > MAX_NUMBER_OF_CLUSTERS:
        n_clusters = MAX_NUMBER_OF_CLUSTERS

    try:
        X_normalized = preprocess_texts(papers)
        model = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=1, max_iter=300)
        y_pred = model.fit_predict(X_normalized)
        plot_clusters_umap(is_hdbscan=False, X=X_normalized, labels=y_pred)

        clusters: Dict[int, list] = {i: [] for i in range(n_clusters)}
        
        for paper_idx, paper in enumerate(papers):
            clusters[int(y_pred[paper_idx])].append(paper)
            
        # Drop clusters that ended up empty
        clusters = {k: v for k, v in clusters.items() if v}

        top_clusters_words = get_top_k_words_from_svd_centroids(
            model.cluster_centers_, top_k=3
        )

        if len(np.unique(y_pred)) >= 2:
            metrics = {
                "SIL": silhouette_score(X_normalized, y_pred, metric="cosine"),
                "DBI": davies_bouldin_score(X_normalized, y_pred),
                "CHI": calinski_harabasz_score(X_normalized, y_pred),
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


def get_hdbscan_params(n_papers):
    """
    Returns the parameters for HDBSCAN based on the number of papers.
    The idea is to avoid having too many clusters for a small len(papers) and more clusters for a larger len(papers).
    """
    if n_papers < 30:
        return 2, 3

    elif n_papers < 50:
        min_cluster_size = max(2, int(0.1 * n_papers))
        min_samples = 3

    elif n_papers < 100:
        min_cluster_size = max(3, int(0.06 * n_papers))
        min_samples = None

    elif n_papers < 200:
        min_cluster_size = int(0.04 * n_papers)
        min_samples = min_cluster_size // 2

    else:
        min_cluster_size = int(0.03 * n_papers)
        min_samples = 5

    logging.info(f"Min size: {min_cluster_size}")
    logging.info(f"Min Sample: {min_samples}")
    return min_cluster_size, min_samples


@st.cache_resource(show_spinner=False)
def get_paper_clusters_hdbscan(papers: List[Paper]):
    """
    Performs HDBSCAN clustering on a list of papers.
    Returns:
        Tuple (clusters_dict, top_words_dict, metrics)
        - clusters_dict: {cluster_id: [list_of_papers]}
        - top_words_dict: {cluster_id: [top_words]}
        - metrics: Dictionary of clustering quality metrics
    """

    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    try:
        X_normalized = preprocess_texts(papers)

        min_cluster_size, min_sample = get_hdbscan_params(len(papers))
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_sample,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = model.fit_predict(X_normalized)

        # Map each HDBSCAN label to the indices of its assigned papers
        paper_indices_by_label: Dict[int, list] = {}
        for paper_idx, label in enumerate(labels):
            paper_indices_by_label.setdefault(label, []).append(paper_idx)

        # ------------------------------------------------------------------ #
        # Cluster dictionary construction
        #
        # Noise points (label == -1) are collected into Cluster 0 so the UI
        # can display them as "Noise Cluster".
        # Real clusters are remapped to contiguous IDs starting at 1.
        # ------------------------------------------------------------------ #
        noise_indices = paper_indices_by_label.get(-1, [])
        clusters: Dict[int, list] = {0: [papers[i] for i in noise_indices]}

        real_labels = sorted(l for l in paper_indices_by_label if l != -1)
        label_to_cid = {label: (rank + 1) for rank, label in enumerate(real_labels)}

        for label in real_labels:
            cid = label_to_cid[label]
            clusters[cid] = [papers[i] for i in paper_indices_by_label[label]]

        # Drop any cluster that ended up empty (e.g. empty noise bucket)
        clusters = {k: v for k, v in clusters.items() if v}

        # ------------------------------------------------------------------ #
        # Keyword extraction
        #
        # Centroids are computed only for real (non-noise) clusters.
        # Cluster 0 receives an empty keyword list.
        # get_top_k_words_from_svd_centroids returns keys 0..K-1, so we
        # remap them to the actual cluster IDs (1..K).
        # ------------------------------------------------------------------ #
        centroids = []
        centroid_cids = []
        for label in real_labels:
            idx = np.array(paper_indices_by_label[label], dtype=int)
            if len(idx) == 0:
                continue
            centroid = np.asarray(X_normalized[idx].mean(axis=0)).ravel()
            centroids.append(centroid)
            centroid_cids.append(label_to_cid[label])

        if len(centroids) == 0:
            return {0: papers}, {}, _DEFAULT_METRICS

        raw_top_words = get_top_k_words_from_svd_centroids(  # keys 0..K-1
            np.vstack(centroids), top_k=3
        )
        top_words = {cid: raw_top_words[i] for i, cid in enumerate(centroid_cids)}  # keys 1..K
        top_words[0] = []

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

        # Build a per-paper label array that uses cluster IDs (not raw HDBSCAN
        # labels) so the PCA plot is consistent with the cluster dictionary.
        plot_labels = np.zeros(len(papers), dtype=int)
        for label in real_labels:
            cid = label_to_cid[label]
            plot_labels[paper_indices_by_label[label]] = cid

        plot_clusters_umap(is_hdbscan=True, X=X_normalized, labels=plot_labels)

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
        legend_label = "Noise" if (cluster == 0 and is_hdbscan) else f"Cluster {cluster + 1}"
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