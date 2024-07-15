
import numpy as np
from sklearn.metrics import roc_curve, auc

def print_tpr_target(fpr, tpr, name, target_fpr):
    indices = None
    for i in range(len(fpr)):
        if fpr[i] >= target_fpr:
            if i == 0:
                indices = [i]
            else:
                indices = [i-1, i]
            break

    if indices is None:
        print(f"{name} TPR at {target_fpr*100}% FPR: {tpr[-1]}. FPR is too high.")
    else:
        tpr_values = [tpr[i] for i in indices]
        print(f"{name} TPR at {target_fpr*100}% FPR: {np.mean(tpr_values) * 100:5.1f}%")

def get_roc(human_scores, machine_scores, max_fpr=1.0):
    fpr, tpr, _ = roc_curve([0] * len(human_scores) + [1] * len(machine_scores), human_scores + machine_scores)
    fpr_auc = [x for x in fpr if x <= max_fpr]
    tpr_auc = tpr[:len(fpr_auc)]
    roc_auc = auc(fpr_auc, tpr_auc)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), float(roc_auc) * (1.0 / max_fpr)