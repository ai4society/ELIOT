import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
import streamlit as st
from sklearn.cluster import KMeans, HDBSCAN
from umap import UMAP
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from arxiv_searcher import Paper
from .preprocessing import (
    preprocess_texts,
    get_top_k_words_from_svd_centroids,
)


MIN_PAPERS_FOR_CLUSTERING = 15
_DEFAULT_METRICS = {"SIL": 0, "DBI": 0, "CHI": 0}

@st.cache_resource(show_spinner=False)
def get_paper_clusters_fuzzy(papers: List[Paper], n_clusters: int = 5):
    """
    Performs Soft Clustering (Fuzzy C-Means) on a list of papers.
    Returns:
        Tuple (clusters_dict, top_words_dict, metrics)
        - clusters_dict: {cluster_id: [list_of_papers]}
        - top_words_dict: {cluster_id: [top_words]}
        - metrics: Dictionary of clustering quality metrics
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, _DEFAULT_METRICS

    if n_clusters > 10:
        n_clusters = 10

    # Minimum membership probability for a secondary cluster assignment.
    threshold = 0.3

    try:
        X_normalized = preprocess_texts(papers)
        X_transposed = X_normalized.T

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_transposed,
            c=n_clusters,
            m=1.7,       # fuzziness parameter
            error=0.005,
            maxiter=1000,
        )

        # ------------------------------------------------------------------ #
        # Cluster assignment (soft)
        #
        # Every paper is guaranteed a place in its best cluster.
        # It is also added to any other cluster whose membership value
        # exceeds the threshold (allow multi-cluster membership).
        # ------------------------------------------------------------------ #
        clusters: Dict[int, list] = {i: [] for i in range(n_clusters)}

        for paper_idx, paper in enumerate(papers):
            membership_values = u[:, paper_idx]
            best_cluster = int(np.argmax(membership_values))

            clusters[best_cluster].append(paper)

            for cluster_idx, prob in enumerate(membership_values):
                if cluster_idx != best_cluster and prob >= threshold:
                    clusters[cluster_idx].append(paper)

        # Drop clusters that ended up empty
        clusters = {k: v for k, v in clusters.items() if v}

        # Hard assignment
        primary_assignment = np.argmax(u, axis=0)

        # ------------------------------------------------------------------ #
        # Centroid recomputation for keyword extraction
        #
        # We intentionally recompute centroids using only hard-assigned papers
        # instead of the soft FCM centroids.  Soft centroids collapse to very
        # similar TF-IDF vectors after SVD projection, causing every cluster
        # to share the same top-k keywords.  Hard-assignment centroids are
        # cluster-specific and yield distinct keyword sets.
        # ------------------------------------------------------------------ #
        hard_centroids = []
        for cluster_id in range(n_clusters):
            primary_indices = np.where(primary_assignment == cluster_id)[0]
            if len(primary_indices) > 0:
                centroid = X_normalized[primary_indices].mean(axis=0)
            else:
                # Fallback use the original FCM centroid when no paper was
                # hard-assigned to this cluster.
                centroid = cntr[cluster_id]
            hard_centroids.append(centroid)

        hard_centroids = np.array(hard_centroids)
        top_clusters_words = get_top_k_words_from_svd_centroids(
            hard_centroids, top_k=3
        )

        y_pred = primary_assignment
        plot_clusters_umap(X_normalized, y_pred)

        metrics = {
            "SIL": silhouette_score(X_normalized, y_pred, metric="cosine"),
            "DBI": davies_bouldin_score(X_normalized, y_pred),
            "CHI": calinski_harabasz_score(X_normalized, y_pred),
        }

        papers_in_multiple_clusters = sum(
            1
            for paper in papers
            if sum(paper in v for v in clusters.values()) > 1
        )

        print(f"Fuzzy Clustering: {len(clusters)} clusters, FPC={fpc:.3f}")
        print(f"Metrics: {metrics}")
        print(f"Top words: {top_clusters_words}")
        print(f"\nTotal: {papers_in_multiple_clusters}/{len(papers)} papers in more than 1 cluster")
        print("=" * 50 + "\n")

        return dict(sorted(clusters.items())), top_clusters_words, metrics

    except Exception as e:
       logging.error(f"Fuzzy Clustering error: {e}")
       return {0: papers}, {}, _DEFAULT_METRICS


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

        clusterer = HDBSCAN(
            min_cluster_size=3,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X_normalized)

        # Map each HDBSCAN label to the indices of its assigned papers
        paper_indices_by_label: Dict[int, list] = {}
        for paper_idx, label in enumerate(labels):
            paper_indices_by_label.setdefault(label, []).append(paper_idx)

        # ------------------------------------------------------------------ #
        # Cluster dictionary construction
        #
        # Noise points (label == -1) are collected into Cluster 0 so the UI
        # can display them as "Uncategorized".
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
        # Centroids are computed only for real (non-noise) clusters;
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
            centroids.append(X_normalized[idx].mean(axis=0))
            centroid_cids.append(label_to_cid[label])

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

        plot_clusters_umap(X_normalized, plot_labels)

        print(f"HDBSCAN Clustering: {len(clusters)} clusters (including noise)")
        print(f"Metrics: {metrics}")
        print(f"Noise Ratio: {noise_ratio}")

        return dict(sorted(clusters.items())), top_words, metrics

    except Exception as e:
       logging.error(f"HDBSCAN Clustering error: {e}")
       return {0: papers}, {}, _DEFAULT_METRICS


def plot_clusters_umap(X, labels, save_path="clusters_umap.png"):
    K = max(1, len(np.unique(labels)))
    n_neighbors = max(2, min(X.shape[0] - 1, X.shape[0] // K))
    
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=42,
    )
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for k in np.unique(labels):
        idx = labels == k
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=40,
            alpha=0.7,
            label=f"Cluster {k}"
        )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("Clusters (UMAP projection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()