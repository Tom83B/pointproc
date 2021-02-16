import numpy as np


def piecewise_apply(func, arr, indices):
    integrated_arr = []
    for ix1, ix2 in zip(indices[:-1], indices[1:]):
        integrated_arr.append(func(arr[ix1:ix2]))

    return np.array(integrated_arr)

def concatenate(lists):
    res = []

    for l in lists:
        for x in l:
            res.append(x)

    return res