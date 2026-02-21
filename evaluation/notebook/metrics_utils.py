import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Metrics for no ground truth available
def calculate_chi(X, y_pred):
    return calinski_harabasz_score(X, y_pred)

def calculate_dbi(X, y_pred):
    return davies_bouldin_score(X, y_pred)

def calculate_silhouette(X, y_pred):
    if len(np.unique(y_pred)) < 2:
        return -1.0
    return silhouette_score(X, y_pred, metric='cosine')

# Metrics for ground truth available
def calculate_nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

def calculate_ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def calculate_accuracy(y_true, y_pred):
    """
    Measures the proportion of correctly assigned samples
    after optimal matching between cluster labels and ground-truth classes via the
    Hungarian algorithm.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)

    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / len(y_true)
