import re
from typing import List

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources():
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

_LEMMATIZER = WordNetLemmatizer()

def clean_and_lemmatize(text: str, use_pos: bool = True) -> str:
    """
    Cleans and lemmatizes the text.

    Args:
        text: text to clean and lemmatize
        use_pos: whether to use POS tagging
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)

    if not use_pos:
        return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)

    tagged = nltk.pos_tag(tokens)
    lemmas = [_LEMMATIZER.lemmatize(w, pos=get_wordnet_pos(tag)) for w, tag in tagged]
    return " ".join(lemmas)


def preprocess_texts(documents: List[str], use_pos: bool = True) -> List[str]:
    """
    Receives texts and returns preprocessed texts (strings)
    """
    return [clean_and_lemmatize(doc, use_pos=use_pos) for doc in documents]