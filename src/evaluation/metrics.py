from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def balanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Downsample majority class before computing confusion matrix."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(42)
    pos_sample = rng.choice(pos_idx, size=n, replace=False)
    neg_sample = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_sample, neg_sample])
    return confusion_matrix(y_true[idx], y_pred[idx])


def per_attack_report(
    errors: np.ndarray,
    y_true: np.ndarray,
    attack_labels: np.ndarray,
    threshold: float,
) -> Dict[str, dict]:
    """Compute Precision/Recall/F1 per attack type.

    Args:
        errors:       Reconstruction error per sample, shape (N,)
        y_true:       Binary labels (0=Normal, 1=Attack), shape (N,)
        attack_labels: String attack type per sample (e.g., 'DDoS', 'Normal'), shape (N,)
        threshold:    Anomaly threshold value

    Returns:
        Dict keyed by attack type with precision/recall/f1/support.
    """
    y_pred = (errors > threshold).astype(int)
    attack_types = [t for t in np.unique(attack_labels) if t != "Normal"]
    report = {}
    for attack in attack_types:
        mask = (attack_labels == attack) | (attack_labels == "Normal")
        rep = classification_report(
            y_true[mask], y_pred[mask], output_dict=True, zero_division=0
        )
        report[attack] = rep.get("1", {})
    return report


def compute_roc_auc(errors: np.ndarray, y_true: np.ndarray) -> float:
    """ROC-AUC score for continuous reconstruction errors vs binary labels."""
    return float(roc_auc_score(y_true, errors))


def print_evaluation(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    attack_labels: Optional[np.ndarray] = None,
) -> None:
    y_pred = (errors > threshold).astype(int)

    print("\n=== Overall Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"], zero_division=0))

    print("=== Balanced Confusion Matrix ===")
    cm = balanced_confusion_matrix(y_true, y_pred)
    print(cm)

    try:
        auc = compute_roc_auc(errors, y_true)
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception:
        pass

    if attack_labels is not None:
        print("\n=== Per-Attack-Type Report ===")
        report = per_attack_report(errors, y_true, attack_labels, threshold)
        for attack, metrics in report.items():
            p = metrics.get("precision", 0)
            r = metrics.get("recall", 0)
            f = metrics.get("f1-score", 0)
            s = metrics.get("support", 0)
            print(f"  {attack:20s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")
