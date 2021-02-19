import numpy as np
from collections import Counter


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


def join_names(*a):
    tmp = concatenate(a)
    _counts = Counter(tmp)
    counts = Counter(tmp)

    res = []
    for x in tmp:
        if _counts[x] > 1:
            i = _counts[x] - counts[x] + 1
            res.append(str(x) + str(i))
            counts[x] -= 1
        else:
            res.append(str(x))

    return res


if __name__ == '__main__':
    nl1 = ['a', 'b', 'c']
    nl2 = ['d', 'b', 'a']

    print(join_names(nl1, nl2))