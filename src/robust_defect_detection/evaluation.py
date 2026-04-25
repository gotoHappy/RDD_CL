import numpy as np


def statistics_from_binary_img_pair(pred_img, target_img, return_details=False):
    pred_img = np.asarray(pred_img).astype(bool)
    target_img = np.asarray(target_img).astype(bool)
    tp_mask = pred_img & target_img
    fp_mask = pred_img & ~target_img
    fn_mask = ~pred_img & target_img
    tn_mask = ~pred_img & ~target_img
    stats = {
        "TP": int(np.sum(tp_mask)),
        "FP": int(np.sum(fp_mask)),
        "FN": int(np.sum(fn_mask)),
        "TN": int(np.sum(tn_mask)),
    }
    if return_details:
        return stats, {"TP": tp_mask, "FP": fp_mask, "FN": fn_mask, "TN": tn_mask}
    return stats


def change_mask_metric(pred_img, target_img):
    stats = statistics_from_binary_img_pair(pred_img, target_img)
    tp = stats["TP"]
    fp = stats["FP"]
    fn = stats["FN"]
    tn = stats["TN"]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    f1_score = 2 * recall * precision / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "iou": iou,
    }
