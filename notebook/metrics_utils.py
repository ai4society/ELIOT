import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import LabelEncoder


def calculate_nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

def calculate_ari(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def calculate_silhouette(X, y_pred):
    if len(np.unique(y_pred)) < 2:
        return -1.0
    
    return silhouette_score(X, y_pred, metric='cosine')

def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)

    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / len(y_true)

