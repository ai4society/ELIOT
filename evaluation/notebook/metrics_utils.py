import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from octis.evaluation_metrics.coherence_metrics import Coherence


# Metrics for no ground truth available
def calculate_chi(X, y_pred):
    return calinski_harabasz_score(X, y_pred)


def calculate_dbi(X, y_pred):
    return davies_bouldin_score(X, y_pred)


def calculate_silhouette(X, y_pred):
    if len(np.unique(y_pred)) < 2:
        return -1.0
    return silhouette_score(X, y_pred, metric='cosine')


def calculate_coherence_cv(topics, corpus, topk=4):
    metric = Coherence(texts=corpus, measure="c_v", topk=topk)
    model_output = {"topics": topics}
    return metric.score(model_output)


def calculate_coherence_npmi(topics, corpus, topk=4):
    metric = Coherence(texts=corpus, measure="c_npmi", topk=topk)
    model_output = {"topics": topics}
    return metric.score(model_output)
