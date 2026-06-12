import numpy as np
import torch


def binary_auc_score(target, scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = target == 1
    n_pos = float(pos.sum())
    n_neg = float(len(target) - pos.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def segmentation_metrics(logits, target, threshold=0.5):
    prob = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    pred = (prob >= threshold).astype(np.uint8)
    gt = target.detach().cpu().numpy().flatten().astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    eps = 1e-8
    auc = binary_auc_score(gt, prob)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "AUC": round(float(auc), 4),
        "F1": round(float(f1), 4),
        "Acc": round(float(acc), 4),
        "Sen": round(float(recall), 4),
        "Spe": round(float(specificity), 4),
        "Pre": round(float(precision), 4),
        "IoU": round(float(iou), 4),
    }
