import os
import time
import shutil

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter

import sod_metric
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # if flatten:
    #     ret = ret.view(-1, ret.shape[-1])

    return ret



from sklearn.metrics import precision_recall_curve


# def calc_f1(y_pred, y_true):
#     batchsize = y_true.shape[0]
#     with torch.no_grad():
#         assert y_pred.shape == y_true.shape
#         f1, auc, P, R = 0, 0, 0, 0
#         y_true = y_true.cpu().numpy()
#         y_pred = y_pred.cpu().numpy()
#         for i in range(batchsize):
#             true = y_true[i].flatten()
#             true = true.astype(np.int)
#             pred = y_pred[i].flatten()
#
#             precision, recall, thresholds = precision_recall_curve(true, pred)
#
#             # auc
#             auc += roc_auc_score(true, pred)
#             # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
#             P += precision
#             R += recall
#             f1 += max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])
#
#     return f1/batchsize, auc/batchsize, P/batchsize, R/batchsize

# def calc_f1(y_pred, y_true):
#     batchsize = y_true.shape[0]
#     with torch.no_grad():
#         assert y_pred.shape == y_true.shape
#         f1_max, P, R, iou, auc = 0.0, 0.0, 0.0, 0.0, 0.0
#         f1_mean = 0.0
#         y_true = y_true.cpu().numpy()
#         y_pred = y_pred.cpu().numpy()
#         for i in range(batchsize):
#             true = y_true[i].flatten()
#             true = true.astype(np.int8)
#             pred = y_pred[i].flatten()
#
#             precision, recall, thresholds = precision_recall_curve(true, pred)
#             auc += roc_auc_score(true, pred)
#
#             inter = (pred * true).sum()
#             union = (pred + true).sum() - inter
#             iou += (inter / union)
#
#             P += precision.mean()
#             R += recall.mean()
#
#             f1_max += max([(2 * p * r) / (p + r + 1e-8) for p, r in zip(precision, recall)])
#
#         f1_max, P, R, iou, auc = f1_max / batchsize, P / batchsize, R / batchsize, iou / batchsize, auc / batchsize
#         f1_mean = (2 * P * R) / (P + R + 1e-8)
#
#     return f1_max, auc, iou, P, R, f1_mean

def calc_f1(y_pred, y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1_max, iou, auc = 0.0, 0.0, 0.0
        precisions, recalls = [], []
        y_true_numpy = y_true.cpu().numpy() # (B, 1, 1024, 1024)
        y_pred_numpy = y_pred.cpu().numpy()

        for i in range(batchsize):
            # DTD methods
            pred_tamper = y_pred[i].squeeze(1)
            target_ = y_true[i].squeeze(1)
            match = (pred_tamper * target_).sum((1, 2))
            preds = pred_tamper.sum((1, 2))
            target_sum = target_.sum((1, 2))
            precisions.append((match / (preds + 1e-8)).mean().item())
            recalls.append((match / (target_sum + 1e-8)).mean().item())

            # print("match", match)
            # print("preds", preds)
            # print("target_sum", target_sum)
            # print("precision", precisions)
            # print("recall", recalls)
            # SAM-Adapter methods
            true = y_true_numpy[i].flatten()
            true = true.astype(np.int8)
            pred = y_pred_numpy[i].flatten()

            precision, recall, thresholds = precision_recall_curve(true, pred)
            auc += roc_auc_score(true, pred)
            inter = (pred * true).sum()
            union = (pred + true).sum() - inter
            iou += (inter / union)

            f1_max += max([(2 * p * r) / (p + r + 1e-8) for p, r in zip(precision, recall)])

        f1_max, iou, auc = f1_max / batchsize, iou / batchsize, auc / batchsize

        precisions = np.array(precisions).mean()
        recalls = np.array(recalls).mean()
        f = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        # print("precisions", precisions)
        # print("recalls", recalls)
        # print("f", f)
        # print("iou", iou)
    return f1_max, auc, iou, precisions, recalls, f
