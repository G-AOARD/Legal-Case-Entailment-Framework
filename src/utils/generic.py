import os
import hashlib
import numpy as np


def merge_override_dict(d, merged_d):
    for k, v in d.items():
        merged_d[k] = v
    return merged_d


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_filename(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def hash(text, length=None):
    hash_text = hashlib.md5(text).hexdigest()
    if length:
        hash_text = hash_text[:length]
    return hash_text


def sort_dict(d, f=lambda x: x[1], reverse=False):
    sorted_d = sorted(d.items(), key=f, reverse=reverse)
    return {x[0]: x[1] for x in sorted_d}


def aggregate_op(a, b, op):
    if op == "max":
        return max(a, b)
    elif op == "sum":
        return a + b
    else:
        raise ValueError(op)


def aggregate_list(x, op):
    if op == "max":
        return max(x)
    elif op == "sum":
        return sum(x)
    else:
        raise ValueError(op)


def top_k_largest_index(matrix, k):
    idx = np.argpartition(matrix.ravel(), matrix.size - k)[-k:]
    return np.column_stack(np.unravel_index(idx, matrix.shape))
