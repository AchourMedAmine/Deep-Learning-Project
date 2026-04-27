import numpy as np
from sklearn.metrics import confusion_matrix
def compute_icbhi_metrics(y_true, y_pred):
    """
    ICBHI official binary evaluation.
    4 classes → binary: Normal (0) vs Abnormal (1,2,3)
    
    Returns: sensitivity, specificity, score
    """
    cm = confusion_matrix(y_true, y_pred)
    # Sensitivity: abnormal samples correctly predicted as ANY abnormal class
    se_num = np.sum(cm[1:, 1:])    # abnormal predicted as abnormal
    se_den = np.sum(cm[1:, :])     # all actual abnormal
    se = se_num / se_den if se_den > 0 else 0
    # Specificity: normal samples correctly predicted as normal
    sp_num = cm[0, 0]              # normal predicted as normal
    sp_den = np.sum(cm[0, :])      # all actual normal
    sp = sp_num / sp_den if sp_den > 0 else 0
    # ICBHI Score
    score = (se + sp) / 2
    return se, sp, score, cm