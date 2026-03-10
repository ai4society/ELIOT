import re
from typing import Dict, List

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

from bertopic.vectorizers import ClassTfidfTransformer

import numpy as np

from arxiv_searcher import Paper


_LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer(min_df=3, max_df=0.7, max_features=3000, ngram_range=(1, 3), stop_words="english")
SVD = TruncatedSVD(n_components=5, random_state=42)

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


def preprocess_texts(papers: List[Paper]):
    """
    Receives texts and returns preprocessed texts (strings)
    """
    clean_text = [clean_and_lemmatize(doc) for doc in [p.title + " " + p.abstract for p in papers]]
    
    X = VECTORIZER.fit_transform(clean_text)
    X = SVD.fit_transform(X)
    X = normalize(X, norm="l2")

    return X


def _remove_substrings(candidates: List[str], top_k: int):
    """
    Filter a ranked list of n-gram candidates by removing entries that are
    substrings of (or contain) another candidate, then return the top_k survivors.

    Example:
        ["neural network", "network", "deep neural network", "learning"]
        -> ["neural network", "learning"]  (with top_k=2)
    """
    final = []
    for word in candidates:
        is_substring = any(
            word != other and (word in other or other in word)
            for other in candidates
        )
        if not is_substring:
            final.append(word)
        if len(final) >= top_k:
            break
    return final


def get_top_k_words_ctfidf(papers, labels, top_k=3):
    texts = [clean_and_lemmatize(p.title + " " + p.abstract) for p in papers]
    unique_labels = sorted(set(labels))

    super_docs = [
        " ".join(texts[i] for i in np.where(labels == lbl)[0])
        for lbl in unique_labels
    ]

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(2, 3),
        max_features=3000
    )
    ctfidf_model = ClassTfidfTransformer()

    X_counts = vectorizer.fit_transform(super_docs)

    X_ctfidf = ctfidf_model.fit_transform(X_counts)
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}

    for i, lbl in enumerate(unique_labels):
        scores = X_ctfidf[i].toarray().ravel()
        ranked_indices = scores.argsort()[::-1]
        candidates = [
            feature_names[j]
            # Get the top_k * X candidates to filter out redundancies.
            # Example:
            #      ["multi agent", "agent", "agents", "multi agents", "planning"]
            #      becomes ["multi agent", "planning"]
            for j in ranked_indices[: top_k * 6]
            if scores[j] > 0
        ]
        top_words[lbl] = _remove_substrings(candidates, top_k)

    return top_words


def get_top_k_words(cntr_svd, top_k=3):
    terms = VECTORIZER.get_feature_names_out()
    top_words = {}

    for i in range(cntr_svd.shape[0]):
        # Reconstructs the weights in the original TF-IDF space
        tfidf_weights = SVD.inverse_transform(cntr_svd[i].reshape(1, -1)).ravel()

        # Get the top_k * X candidates to filter out redundancies.
        # Example:
        #      ["multi agent", "agent", "agents", "multi agents", "planning"]
        #      becomes ["multi agent", "planning"]
        candidate_indices = tfidf_weights.argsort()[-(top_k * 5):][::-1]
        candidates = [terms[idx] for idx in candidate_indices]
        top_words[i] = remove_substrings(candidates, top_k)

    return top_words
