from typing import List, Dict
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from arxiv_searcher import Paper
import re
import nltk
from nltk.stem import WordNetLemmatizer


# Setup NLTK (Download wordnet if not present)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=1000, 
    max_df=0.65,
    min_df=3,
    ngram_range=(1, 2),
    strip_accents='unicode'
)


def clean_text(text: str) -> str:
    """
    Applies basic essential text cleaning AND lemmatization:
        1. Lowercase
        2. Remove single characters (noise)
        3. Lemmatize (convert words to their base form)
        
    Args:
        text: Raw input text

    Returns:
        Cleaned and lemmatized text string
    """
    # Lowercase first for better lemmatization matches
    text = text.lower()
    
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    # Remove single characters (e.g. " a ", " b ")
    text = re.sub(r'\b[a-z]\b', '', text)

    # Lemmatize tokens
    tokens = text.split()
    # Lemmatize twice: once for verbs (running->run) and once for nouns (cars->car)
    tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in tokens]
    
    return ' '.join(tokens)


def preprocess_and_vectorize(papers: List[Paper], n_components: int = 100):
    """
    Converts paper texts into normalized semantic vectors using TF-IDF and SVD.
    
     Args:
        papers: List of Paper objects containing title and abstract
        n_components: Number of latent semantic dimensions
    
    Returns:
        X_normalized: Reduced and normalized document embeddings
        svd: Fitted TruncatedSVD model
        vectorizer: Fitted TF-IDF vectorizer
    """
    if not papers:
        return None, None, None

    documents = [clean_text(f"{p.title} {p.abstract}") for p in papers]
    
    X_tfidf = vectorizer.fit_transform(documents)

    # SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    
    # Normalize for cosine distance
    X_normalized = normalize(X_reduced, norm="l2")

    return X_normalized, svd, vectorizer


def get_top_k_words_from_svd_centroids(cntr_svd, svd, vectorizer, top_k=3):
    """
    Extracts the main words from each centroid in SVD space.
    
    Args:
        cntr_svd: Centroids in reduced SVD space (n_clusters, n_components)
        svd: TruncatedSVD fitted
        vectorizer: TfidfVectorizer fitted
        top_k: Number of words by cluster
    
    Returns:
        Dictionary {cluster_id: [top_k_words]}
    """
    terms = vectorizer.get_feature_names_out()
    top_words = {}

    for i in range(cntr_svd.shape[0]):
        # Reconstructs the weights in the original TF-IDF space
        tfidf_weights = svd.inverse_transform(cntr_svd[i].reshape(1, -1)).ravel()

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


def get_top_k_words(centroids, top_k=3):
    """
    Returns the top k words for each cluster based on centroid values.
    
    Args:
        centroids: Cluster centroids from KMeans (shape: n_clusters x n_features)
        top_k: Number of top words to extract per cluster
    
    Returns:
        Dictionary mapping cluster_id -> list of top k words
    """
    top_words = {}
    terms = vectorizer.get_feature_names_out()
    
    for i in range(len(centroids)):
        # Get indices of top features with highest values
        # We fetch more candidates initially to allow for filtering redundancy
        candidate_indices = centroids[i].argsort()[-(top_k * 3):][::-1]
        candidates = [terms[idx] for idx in candidate_indices]
        
        # Filter out substrings (e.g., remove "agent" if "multi agent" is present)
        final_words = []
        for word in candidates:
            # Keep a word only if it is NOT a substring of another candidate 
            # We use regex \b to ensure we match whole words (e.g. avoid matching "ear" in "learning")

            is_substring = False
            for other_word in candidates:
                if word != other_word:
                    # Check if 'word' is a whole-word substring of 'other_word'
                    if re.search(r'\b' + re.escape(word) + r'\b', other_word):
                        is_substring = True
                        break
            
            if not is_substring:
                final_words.append(word)
                
            if len(final_words) >= top_k:
                break
                
        top_words[i] = final_words
    
    return top_words