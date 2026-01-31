import logging
from typing import List
import streamlit as st
from sklearn.cluster import KMeans
from arxiv_searcher import Paper
from preprocessing import preprocess_and_vectorize, get_top_k_words

MIN_PAPERS_FOR_CLUSTERING = 10

@st.cache_resource(show_spinner=False)
def get_paper_clusters(papers: List[Paper], n_clusters: int = 1):
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
