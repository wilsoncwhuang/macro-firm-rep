from unittest import result
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import tqdm
import pandas as pd

class AUCMetric:
    def __init__(self, horizons):
        self.horizons = horizons
        self.y_trues = []
        self.y_preds = []
        self.fallback = 0.5  # Fallback AUC value in case of error

    def update_state(self, y_pred, y_true):
        # Convert to numpy for compatibility with scikit-learn's roc_auc_score
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        default_event = 1
        defaults = y_true_np == default_event
        default_label = defaults
        y_pred_np = y_pred_np[:, :, 1]  # Get the probability of the default event

        self.y_trues.append(default_label)
        self.y_preds.append(y_pred_np)
    
    def safe_roc_auc(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # need at least two classes
        if np.unique(y_true).size < 2:
            return self.fallback, 1
        try:
            return float(roc_auc_score(y_true, y_pred)), 0
        except Exception as e:
            return self.fallback, 1

    def result(self):
        pbar = tqdm.tqdm(
            self.horizons, desc="Computing AUC for horizon", leave=False, ncols=50
        )
        results = {}
        for horizon in pbar:
            pbar.set_description(f"Computing AUC for horizon: {horizon}")
            try:
                if horizon == "all":
                    y_true_flat = np.concatenate(self.y_trues)
                    y_pred_flat = np.concatenate(self.y_preds)
                    aucs = []
                    for h in range(y_true_flat.shape[1]):
                        auc_h, flag = self.safe_roc_auc(y_true_flat[:, h], y_pred_flat[:, h])
                        if flag == 0:
                            aucs.append(auc_h)
                    auc = float(np.mean(aucs))
                else:
                    horizon_index = horizon - 1
                    y_true_flat = np.concatenate(
                        [array[:, horizon_index] for array in self.y_trues]
                    )
                    y_pred_flat = np.concatenate(
                        [array[:, horizon_index] for array in self.y_preds]
                    )
                    auc, _ = self.safe_roc_auc(y_true_flat, y_pred_flat)
                results[horizon] = auc
            except Exception as e:
                results[horizon] = self.fallback
        return results

    def reset_state(self):
        self.y_trues = []
        self.y_preds = []

def auc(x, y, reorder=False):
    """TODO: Docstring for auc.
    :returns: TODO
    """
    direction = 1
    if reorder:
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <=0):
                direction = -1
            else:
                raise ValueError("x array is not increasing")

    area = direction * np.trapz(y, x)

    return area


def cap_curve(y_true, y_score):
    """TODO: Docstring for CAP_curve.
    Parameters
    -----
    y_true: array, shape = [n_samples]
    y_score: array, shape = [n_samples]
    :returns: TODO
    """
    pos_label = 1
    # y_true: boolean vector
    y_true = (y_true == pos_label)
    # sort scores and corresponding truth values
    desc_score_indicies = np.argsort(y_score, kind="mergesort")[::-1]

    y_score = y_score[desc_score_indicies]
    y_true = y_true[desc_score_indicies]


    # accumulate true-positive
    tps = np.cumsum(y_true)
    # accumulate total
    totals = np.cumsum(np.ones(y_true.shape))

    if tps[-1] == 0:
        tpr = np.zeros_like(tps, dtype=float)
    else:
        tpr = tps / tps[-1]
    totalr = totals / totals[-1]

    return tpr, totalr


def cap_ar_score(y_true, y_score):
    """TODO: Docstring for cap_ar_score.
    :returns: TODO
    """
    tpr_m, totalr_m = cap_curve(y_true, y_score)
    tpr_p, totalr_p = cap_curve(y_true, y_true)
    auc_m = auc(totalr_m, tpr_m)
    auc_p = auc(totalr_p, tpr_p)
    auc_r = auc(totalr_p, totalr_p)
    ar = (auc_m - auc_r)/(auc_p - auc_r)

    return ar
    
def agg_rmse(y_true, y_pred, norm_factor=1, factor=1):
    """
    args:
        y_pred: (np.array)
        y_true: (np.array)
        norm_factor: (np.array or constant)
    """
    # only keep items not inf
    # because y_true could be zeros in some months
    if np.sum(norm_factor) == 0.0:
        norm_factor = 1
    scores = np.setdiff1d(
            (y_pred - y_true)**2 / (norm_factor)**2,
             np.inf) * (factor ** 2)
    return np.sqrt(np.mean(scores))

def agg_rmsne(y_true, y_pred, norm_factor=1, factor=1):
    """
    args:
        y_pred: (np.array)
        y_true: (np.array)
        norm_factor: (np.array or constant)
    """
    if np.sum(norm_factor) == 0.0:
        norm_factor = 1

    # avoid division by zero for y_true == 0
    denom = np.clip(y_true, 1e-12, None)
    scores = np.setdiff1d(
        ((y_pred - y_true) / denom) ** 2 / (norm_factor ** 2),
        np.inf
    ) * (factor ** 2)

    return np.sqrt(np.mean(scores))

def agg_rmsle(y_true, y_pred, factor=1):
    """
    args:
        y_pred: (np.array)
        y_true: (np.array)
    """
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    errors = (np.log1p(y_pred) - np.log1p(y_true)) ** 2
    return np.sqrt(np.mean(errors)) * factor

def r_square(y_true, y_pred):
    """
    args:
        y_pred: (np.array)
        y_true: (np.array)
    """
    # only keep items not inf
    # because y_true could be zeros in some months
    
    y_true_mean = np.mean(y_true)
    ss_total = np.sum(np.square(y_true - y_true_mean))
    ss_res = np.sum(np.square(y_true - y_pred))
    
    r_square = 1- ss_res/ss_total
    
    return r_square