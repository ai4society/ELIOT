import logging
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from arxiv_searcher import Paper
from preprocessing import preprocess_and_vectorize, get_top_k_words

MIN_PAPERS_FOR_CLUSTERING = 10
MEMBERSHIP_THRESHOLD = 0.2

@st.cache_resource(show_spinner=False)
def get_paper_clusters(papers: List[Paper], n_clusters: int = 5):
    """
    Perform clustering on a list of papers.
    
    Returns:
        Tuple of (clusters_dict, top_words_dict)
        - clusters_dict: {cluster_id: [papers]}
        - top_words_dict: {cluster_id: [top_words]}
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}
    
    if n_clusters > 10:
        n_clusters = 10

    try:
        X = preprocess_and_vectorize(papers)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        top_clusters_words = get_top_k_words(kmeans.cluster_centers_, top_k=3)
        clusters = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(papers[idx])
        
        # Sort clusters by label keys to ensure consistent order
        sorted_clusters = dict(sorted(clusters.items()))
        return sorted_clusters, top_clusters_words
    except Exception as e:
        # Log error in console but return a fallback
        print(f"Clustering error: {e}")
        logging.error(f"Clustering error: {e}")
        return {0: papers}, {}


@st.cache_resource(show_spinner=False)
def get_paper_clusters_fuzzy(papers: List[Paper], n_clusters: int = 5):
    """
    Performs Soft Clustering (Fuzzy C-Means) on a list of papers.

    Returns:
        Tuple (clusters_dict, top_words_dict)
        - clusters_dict: {cluster_id: [list_of_papers]} (an article can be in more than 1 clusters)
        - top_words_dict: {cluster_id: [top_words]}
    """
    if not papers or len(papers) < MIN_PAPERS_FOR_CLUSTERING:
        return {0: papers}, {}

    if n_clusters > 10: n_clusters = 10

    try:
        X_sparse = preprocess_and_vectorize(papers)
        X_dense = X_sparse.toarray().T 
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_dense, c=n_clusters, m=2, error=0.005, maxiter=1000
        )

        top_clusters_words = get_top_k_words(cntr, top_k=3)
        clusters = {i: [] for i in range(n_clusters)}
        
        for paper_idx in range(len(papers)):
            membership_values = u[:, paper_idx]
            
            for cluster_idx, prob in enumerate(membership_values):
                if prob >= MEMBERSHIP_THRESHOLD:
                    clusters[cluster_idx].append(papers[paper_idx])

        # Remove clusters that may have become empty due to the threshold
        clusters = {k: v for k, v in clusters.items() if len(v) > 0}
        
        return dict(sorted(clusters.items())), top_clusters_words

    except Exception as e:
        logging.error(f"Fuzzy Clustering error: {e}")
        return {0: papers}, {}