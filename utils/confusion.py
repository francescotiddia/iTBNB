from math import sqrt
import numpy as np

def _safe_div(n, d):
    """Safe division: returns 0 if denominator is 0."""
    return n / d if d != 0 else 0


def fpr(fp, tn):
    """False Positive Rate = FP / (FP + TN)"""
    return _safe_div(fp, fp + tn)


def tpr(tp, fn):
    """True Positive Rate (Recall) = TP / (TP + FN)"""
    return _safe_div(tp, tp + fn)


def fnr(tp, fn):
    """False Negative Rate (Type II error) = FN / (TP + FN)"""
    return _safe_div(fn, tp + fn)


def tnr(tn, fp):
    """True Negative Rate (Specificity) = TN / (TN + FP)"""
    return _safe_div(tn, tn + fp)


def precision(tp, fp):
    """Precision = TP / (TP + FP)"""
    return _safe_div(tp, tp + fp)

def mcc(tp, fp, tn, fn):
    """Matthews Correlation Coefficient"""
    tp = np.float64(tp)
    fp = np.float64(fp)
    tn = np.float64(tn)
    fn = np.float64(fn)

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if den <= 0:
        return 0.0

    return num / np.sqrt(den)

def f1_score(tp, fp, fn):
    """F1 = 2 * Precision * Recall / (Precision + Recall)"""
    prec = precision(tp, fp)
    rec = tpr(tp, fn)
    return _safe_div(2 * prec * rec, prec + rec)


def me(tp, fp, tn, fn):
    """Misclassification Error = (FP + FN) / total"""
    total = tp + fp + tn + fn
    return _safe_div(fp + fn, total)


def acc(tp, fp, tn, fn):
    """Accuracy = (TP+TN)/(TP+TN+FP+FN)"""
    return _safe_div(tp + tn, tp + tn + fp + fn)


def all_metrics(tp, fp, tn, fn):
    """Compute common binary classification metrics"""
    return {
        'precision': precision(tp, fp),
        'recall': tpr(tp, fn),
        'specificity': tnr(tn, fp),
        'fpr': fpr(fp, tn),
        'f1': f1_score(tp, fp, fn),
        'mcc': mcc(tp, fp, tn, fn),
        'misclassification_error': me(tp, fp, tn, fn),
        'fnr': fnr(tp, fn),
        'accuracy': acc(tp, fp, tn, fn)
    }


def confusion_matrix_np(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y = 2 * y_true + y_pred
    y = np.bincount(y, minlength=4)
    y = y.reshape(2, 2)
    return y


