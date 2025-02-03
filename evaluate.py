import numpy as np
from sklearn.metrics import roc_curve, auc, log_loss

def top100_auc(heatmap_path_li, y_true):
    """
    Computes AUC based on the top 100 highest values from each heatmap.

    Args:
        heatmap_path_li (list): List of file paths to heatmap `.npy` files.
        y_true (list or array): Ground truth binary labels (0 or 1).

    Returns:
        float: AUC score.
    """
    score_list = []
    for heatmap_path in heatmap_path_li:
        heatmap = np.load(heatmap_path)

        heatmap_nz = heatmap[heatmap!=0]
        sorted_nz = np.sort(heatmap_nz)[::-1]

        if len(sorted_nz) > 100:
            top_100 = sorted_nz[:100]
        else:
            top_100 = sorted_nz

        score_list.append(np.mean(top_100))

    score_list = [0 if np.isnan(num) else num for num in score_list]
    fpr, tpr, ths = roc_curve(y_true, score_list)
    model_auc = auc(fpr, tpr)
    
    return model_auc


def negative_log_likelihood(y_true, y_prob, eps=1e-15):
    """
    Computes Negative Log-Likelihood (NLL) loss.

    Args:
        y_true (list or array): Ground truth binary labels (0 or 1).
        y_prob (list or array): Predicted probabilities.
        eps (float): Small value to prevent log(0) errors.

    Returns:
        float: Log loss value (negative log-likelihood).
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)

    return log_loss(y_true, y_prob)