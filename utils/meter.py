import numpy as np
from sklearn.metrics import roc_curve, accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class ROCMeter(object):
    """Compute TPR with fixed FPR"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)

    def update(self, target, output):
        # If we use cross-entropy
        if len(output.shape) > 1 and output.shape[1] > 1:
            output = output[:,1]
        elif len(output.shape) > 1 and output.shape[1] == 1:
            output = output[:,0]
        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output])

    def get_tpr(self, fixed_fpr):
        fpr, tpr, thr = roc_curve(self.target, self.output)
        tpr_filtered = tpr[fpr <= fixed_fpr]
        if len(tpr_filtered) == 0:
            return 0.0
        return tpr_filtered[-1]

    def get_accuracy(self, thr=0.5):
        acc = accuracy_score(self.target,
                             self.output >= thr)
        return acc

    def get_top_hard_examples(self, top_n=10):
        diff_arr = np.abs(self.target - self.output)
        hard_indexes = np.argsort(diff_arr)[::-1]
        hard_indexes = hard_indexes[:top_n]
        return hard_indexes, self.target[hard_indexes], self.output[hard_indexes]
        
       