
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Union


try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_and_lemmatize(text: str) -> str:
    """
    Performs basic cleaning and lemmatization of the text with POS Tagging.
    """
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    tokens = nltk.word_tokenize(text)
    
    # POS Tagging
    # Returns list of tuples (word, tag)
    # Ex: [('running', 'VBG'), ('fast', 'RB')]
    tagged_tokens = nltk.pos_tag(tokens)
    
    # Contextual Lemmatization
    lemmatized_tokens = []
    for word, tag in tagged_tokens:
        wntag = get_wordnet_pos(tag)
        # Lemmatize with the correct POS
        lemma = lemmatizer.lemmatize(word, pos=wntag)
        lemmatized_tokens.append(lemma)
    
    return " ".join(lemmatized_tokens)

def preprocess_corpus(
    documents: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: float = 0.7,
    max_features: int = 1000
):
    print("Starting lemmatization and cleaning...")
    preprocessed_docs = [clean_and_lemmatize(doc) for doc in documents]
    
    print("Starting vectorization (Tokenization, Stopwords, Pruning)...")
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )
    
    X = vectorizer.fit_transform(preprocessed_docs)
    
    return X

