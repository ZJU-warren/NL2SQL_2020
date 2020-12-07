import numpy as np


def acc(pred, gt, top: int = 1):
    total = pred.shape[0]
    shot = 0
    for _ in range(total):
        if type(top) == int:
            n_pred = pred[_].argsort()[::-1][0:top]

            if top == 1:
                if n_pred[0] == gt[_]:
                    shot += 1
                continue

        else:
            n_pred = pred[_].argsort()[::-1][0:top[_]]

        if set(n_pred) == set(gt[_]):
            shot += 1

    return shot / total
