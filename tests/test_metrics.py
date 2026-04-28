import numpy as np
from src.evaluation.metrics import balanced_confusion_matrix, compute_roc_auc


def test_balanced_confusion_matrix_shape():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 900 + [1] * 100)
    y_pred = rng.integers(0, 2, size=len(y_true))
    cm = balanced_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    # Both classes should contribute equally
    assert cm.sum() == 200  # 100 from each class


def test_roc_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    errors = np.array([0.1, 0.2, 0.9, 0.95])
    auc = compute_roc_auc(errors, y_true)
    assert auc == 1.0


def test_roc_auc_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    errors = rng.uniform(0, 1, size=200)
    auc = compute_roc_auc(errors, y_true)
    assert 0.0 <= auc <= 1.0
