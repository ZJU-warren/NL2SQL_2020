import numpy as np


def predict_generate(pred, top: int = 1):
    total = pred.shape[0]
    result = []
    for _ in range(total):
        if type(top) == int:
            n_pred = pred[_].argsort()[::-1][0:top]
            if top == 1:
                n_pred = n_pred[0]
        else:
            n_pred = pred[_].argsort()[::-1][0:top[_]]
        result.append(n_pred)
    return result
