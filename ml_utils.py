import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from arxiv_searcher import Paper
from preprocessing import (
    preprocess_and_vectorize,
    get_top_k_words,
    get_top_k_words_from_svd_centroids,
)


MIN_PAPERS_FOR_CLUSTERING = 10


@st.cache_resource(show_spinner=False)
def get_paper_clusters_fuzzy(papers: List[Paper], n_clusters: int = 5):
    """
    Performs Soft Clustering (Fuzzy C-Means) on a list of papers.
    Returns:
        Tuple (clusters_dict, top_words_dict)
        - clusters_dict: {cluster_id: [list_of_papers]}
        - top_words_dict: {cluster_id: [top_words]}
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}, {}
    
    if n_clusters > 10: 
        n_clusters = 10
    
    #threshold = min(1.5 * (1 / n_clusters), 0.3)
    threshold = 0.3

    try:
        X_normalized, svd, vec = preprocess_and_vectorize(papers, n_components=100)
        
        if X_normalized is None:
            return {0: papers}, {}, {}

        X_transposed = X_normalized.T

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_transposed, 
            c=n_clusters, 
            m=1.7,  # Fuzziness parameter
            error=0.005, 
            maxiter=1000, 
            metric="cosine"
        )

        # Assign papers to clusters
        clusters = {i: [] for i in range(n_clusters)}
        
        for paper_idx in range(len(papers)):
            membership_values = u[:, paper_idx]
            
            # Always assign to best cluster
            best_cluster = int(np.argmax(membership_values))
            clusters[best_cluster].append(papers[paper_idx])
            
            # Also assign to other clusters if membership is high enough
            for cluster_idx, prob in enumerate(membership_values):
                if cluster_idx != best_cluster and prob >= threshold:
                    clusters[cluster_idx].append(papers[paper_idx])
        
        # Remove empty clusters
        clusters = {k: v for k, v in clusters.items() if len(v) > 0}

        # Identify hard (primary) cluster assignment for each paper
        primary_assignment = np.argmax(u, axis=0) # Cluster with the highest membership for each paper
        
        # This next loop avoids have same top k words equal for all clusters 
    
        # We recompute centroids using only hard assignments to obtain
        # cluster-specific representations for keyword extraction.
        # Soft FCM centroids collapsed to very similar TF-IDF vectors after SVD inversion,
        # causing all clusters to share the same top-k words.

        # Recalculate centroid for each cluster using only its hard assignments
        new_centroids = []
        for cluster_id in range(n_clusters):
            # Index of the primary papers in this cluster
            primary_indices = np.where(primary_assignment == cluster_id)[0]
            
            if len(primary_indices) > 0:
                cluster_vectors = X_normalized[primary_indices]
                new_centroid = cluster_vectors.mean(axis=0)
                new_centroids.append(new_centroid)
            else:
                # Fallback use Fuzzy's original centroid
                new_centroids.append(cntr[cluster_id])
        
        new_centroids = np.array(new_centroids)
        
        top_clusters_words = get_top_k_words_from_svd_centroids(
            new_centroids, svd, vec, top_k=3
        )

        metrics = {}
        y_pred = primary_assignment
        plot_clusters_pca(X_normalized, y_pred)

        metrics["SIL"] = silhouette_score(
            X_normalized,
            y_pred,
            metric="cosine"
        )
        metrics["DBI"] = davies_bouldin_score(
            X_normalized,
            y_pred
        )
        metrics["CHI"] = calinski_harabasz_score(
            X_normalized,
            y_pred
        )

        print(f"Fuzzy Clustering: {len(clusters)} clusters, FPC={fpc:.3f}")
        print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")
        print(f"Metrics: {metrics}")
        print(f"Top words: {top_clusters_words}")

        count_multi = 0
        for paper_idx in range(len(papers)):
            paper = papers[paper_idx]
            num_clusters = sum(1 for cluster_papers in clusters.values() if paper in cluster_papers)
            if num_clusters > 1:
                count_multi += 1

        print(f"\nTotal: {count_multi}/{len(papers)} papers in more than 1 cluster")
        print("=" * 50 + "\n")

        return dict(sorted(clusters.items())), top_clusters_words, metrics
        
    except Exception as e:
        logging.error(f"Fuzzy Clustering error: {e}")
        return {0: papers}, {}, {}


def plot_clusters_pca(X, labels, save_path="clusters_pca.png"):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

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

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters (PCA projection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()