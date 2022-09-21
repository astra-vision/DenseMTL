# Adapted from https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth/blob/master/evaluation/metrics.py
# original version: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np


class SemanticMetrics(object):
    def __init__(self, n_classes):
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
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        self.mIoU = mean_iu

        return dict(Acc=acc, mIoU=mean_iu), cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_scores(self):
        score, class_iou = self._metrics()
        metrics = {f'{k}': v for k, v in score.items()}
        class_m = {f'cls_{k:02d}': v for k, v in class_iou.items()}
        self.iou = class_m
        return {**metrics, **class_m}
