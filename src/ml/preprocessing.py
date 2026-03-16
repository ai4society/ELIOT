import re
from typing import List

import nltk
import numpy as np
from bertopic.vectorizers import ClassTfidfTransformer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP

from arxiv_searcher import Paper


_LEMMATIZER = WordNetLemmatizer()
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DIMENSIONALITY_REDUCER = UMAP(n_neighbors=15, n_components=10, metric="cosine", random_state=42)
CLASS_BASED_TFIDF = ClassTfidfTransformer()

resources = [
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("tokenizers/punkt", "punkt"),
]
for path, pkg in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)


def embed_and_normalize(texts):
    X = MODEL.encode(texts, show_progress_bar=False)
    X = normalize(X, norm="l2")
    return X


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_and_lemmatize(text: str) -> str:
    """
    Cleans and lemmatizes the text.

    Args:
        text: text to clean and lemmatize
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)

    tagged = nltk.pos_tag(tokens)
    lemmas = [_LEMMATIZER.lemmatize(w, pos=get_wordnet_pos(tag)) for w, tag in tagged]
    return " ".join(lemmas)


def preprocess_texts(papers: List[Paper]) -> List[str]:
    """Cleans and lemmatizes texts. Returns preprocessed strings."""
    return [clean_and_lemmatize(doc) for doc in [p.title + " " + p.abstract for p in papers]]


def embed_and_reduce(cleaned_texts: List[str]) -> np.ndarray:
    """Embeds texts and reduces dimensionality for clustering."""
    X = embed_and_normalize(cleaned_texts)
    X = DIMENSIONALITY_REDUCER.fit_transform(X)
    return X


def _remove_substrings(candidates: List[str], top_k: int) -> List[str]:
    """
    Filter a ranked list of n-gram candidates by removing entries that are 
    strictly substrings of another longer candidate, preserving the most 
    specific terms. Returns the top_k survivors.

    Example:
        ["neural network", "network", "deep neural network", "learning"]
        -> ["deep neural network", "learning"]  (with top_k=2)
    """
    final = []
    for word in candidates:
        is_substring = any(
            word != other and word in other for other in candidates
        )
        if not is_substring:
            final.append(word)

        if len(final) >= top_k:
            break

    return final


def get_cluster_keywords(cleaned_texts: List[str], labels: List[int], top_k: int = 4) -> dict:
    """
    Extracts the top-k representative n-gram keywords for each cluster using c-TF-IDF.

    Expects the same preprocessed texts used for clustering (output of `preprocess_texts`),

    Args:
        cleaned_texts: List of lemmatized/cleaned text strings, one per document.
                     Must be the same texts used to produce the cluster `labels`.
        labels:      Cluster label for each document (same order as `cleaned_texts`).
        top_k:       Number of top keywords to return per cluster. Defaults to 4.

    Returns:
        A dict mapping each unique label to a list of up to `top_k` keyword strings.
        Example: {0: ["neural network", "deep learning"], 1: ["image segmentation"]}
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))

    super_docs = [
        " ".join(cleaned_texts[i] for i in np.where(labels == lbl)[0])
        for lbl in unique_labels
    ]

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=3000,
        max_df=0.7,
        ngram_range=(2, 3),
    )

    X_counts = vectorizer.fit_transform(super_docs)

    X_ctfidf = CLASS_BASED_TFIDF.fit_transform(X_counts)
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}

    for i, lbl in enumerate(unique_labels):
        scores = X_ctfidf[i].toarray().ravel()
        ranked_indices = scores.argsort()[::-1]
        candidates = [
            feature_names[j]
            for j in ranked_indices[: top_k * 10]
            if scores[j] > 0
        ]
        top_words[lbl] = _remove_substrings(candidates, top_k)

    return top_words