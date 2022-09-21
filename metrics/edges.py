import numpy as np


class EdgesMetrics(object):
    def __init__(self, n_classes=2):
        self.mIoU = 0
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def measure(self, label_trues, label_preds):
        label_trues = label_trues.detach().cpu().numpy()
        label_preds = label_preds.detach().cpu().numpy()
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def _metrics(self):
        hist = self.confusion_matrix
        f1 = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        mf1 = np.nanmean(f1)
        return mf1

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_scores(self):
        self.f1 = self._metrics()
        return {'F1': self.f1}
