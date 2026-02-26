import re
from typing import Dict, List

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from arxiv_searcher import Paper


_LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer(min_df=3, max_df=0.7, max_features=3000, ngram_range=(1, 3))
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
    print(f"SVD explained variance: {SVD.explained_variance_ratio_.sum():.2%}")

    return X


def get_top_k_words_from_svd_centroids(cntr_svd, top_k=3):
    """
    Extracts the main words from each centroid in SVD space.
    
    Args:
        cntr_svd: Centroids in reduced SVD space (n_clusters, n_components)
        top_k: Number of words by cluster
    
    Returns:
        Dictionary {cluster_id: [top_k_words]}
    """
    terms = VECTORIZER.get_feature_names_out()
    top_words = {}

    for i in range(cntr_svd.shape[0]):
        # Reconstructs the weights in the original TF-IDF space
        tfidf_weights = SVD.inverse_transform(cntr_svd[i].reshape(1, -1)).ravel()

        # Get the top_k * 3 candidates to filter out redundancies.
        # Example:
        #      ["multi agent", "agent", "agents", "multi agents", "planning"]
        #      becomes ["multi agent", "planning"]
        candidate_indices = tfidf_weights.argsort()[-(top_k * 3):][::-1]
        candidates = [terms[idx] for idx in candidate_indices]

        final_words = []
        for word in candidates:
            is_substring = False
            for other_word in candidates:
                if word != other_word:
                    # Checks if 'word' is a substring of 'other_word'
                    if re.search(r'\b' + re.escape(word) + r'\b', other_word):
                        is_substring = True
                        break
            
            if not is_substring:
                final_words.append(word)
                
            if len(final_words) >= top_k:
                break
        
        top_words[i] = final_words

    return top_words