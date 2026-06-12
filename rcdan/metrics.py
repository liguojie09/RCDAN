import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def segmentation_metrics(logits, target, threshold=0.5):
    prob = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    pred = (prob >= threshold).astype(np.uint8)
    gt = target.detach().cpu().numpy().flatten().astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    eps = 1e-8
    auc = roc_auc_score(gt, prob) if len(np.unique(gt)) > 1 else float("nan")
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
