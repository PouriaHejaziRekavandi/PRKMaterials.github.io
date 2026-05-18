import numpy as np
from scipy.spatial.distance import cdist

def to_array(x):
    if hasattr(x, "data"):
        x = np.asarray(x.data)
    else:
        x = np.asarray(x)
    return x if x.ndim == 2 else x[:, None]

def multisource_metrics(gt, pr, pos_mm, neighbors):
    def get_local_maxima(vec):
        vec = vec.ravel()
        gmax = np.max(vec)
        if gmax <= 0: return np.array([np.argmax(vec)])
        cand = [i for i in range(len(vec)) if len(neighbors[i]) > 0 and np.all(vec[i] > vec[neighbors[i]])]
        if not cand: return np.array([np.argmax(vec)])
        cand = sorted(cand, key=lambda i: vec[i], reverse=True)
        selected = []
        for i in cand:
            if not selected: selected.append(i)
            else:
                d = np.linalg.norm(pos_mm[selected] - pos_mm[i], axis=1)
                if np.all(d > 30.0): selected.append(i)
        return np.array(selected)

    gt_max = get_local_maxima(gt)
    pr_max = get_local_maxima(pr)
    D = cdist(pos_mm[gt_max], pos_mm[pr_max])
    min_d = D.min(axis=1)
    return np.mean(min_d), np.mean(min_d <= 30.0) * 100

def median_mad(data):
    arr = np.array(data)
    arr = arr[np.isfinite(arr)] # حذف مقادیر نامعتبر
    if len(arr) == 0:
        return np.nan, np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return med, mad
