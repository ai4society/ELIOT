from typing import List, Dict
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
    max_features=100, 
    max_df=0.20,
    min_df=2,
    ngram_range=(1, 2),
    strip_accents='unicode'
)

def clean_text(text: str) -> str:
    """
    Applies basic essential text cleaning AND lemmatization:
    1. Lowercase
    2. Remove single characters (noise)
    3. Lemmatize (convert words to their base form)
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

def preprocess_and_vectorize(papers: List[Paper]):
    """
    Extracts text from papers and vectorizes them using TF-IDF.
    Returns the vector matrix.
    """
    if not papers:
        return None

    # Apply cleaning
    documents = [clean_text(f"{p.title} {p.abstract}") for p in papers]
    
    # Normalizing makes it use cosine distance equivalent
    return normalize(vectorizer.fit_transform(documents), norm='l2')


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