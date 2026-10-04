import numpy as np
from scipy.sparse.csgraph import connected_components


def one_hot(ans, CHOICES):
    """One-hot vector of an answer letter; all zeros if unanswered or not a valid choice."""
    v = np.zeros(len(CHOICES), dtype=int)
    if isinstance(ans, str) and ans.strip().upper() in CHOICES:
        v[np.argmax(CHOICES == ans.strip().upper())] = 1
    return v


def similarity(a, b):
    """1 if same answer. Two non-answers are NOT similar."""
    if a.sum() == 0 or b.sum() == 0:
        return 0
    return int(np.array_equal(a, b))


def aggregate(answers, CHOICES, req_votes):
    """
    PVA
    """
    outs = [one_hot(a, CHOICES) for a in answers]
    n = len(outs)

    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = similarity(outs[i], outs[j])

    _, comp = connected_components(A, directed=False)

    best, size = None, 0
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        if outs[idx[0]].sum() == 0:      # skip unanswered singletons
            continue
        if len(idx) > size:
            best, size = idx, len(idx)

    if best is None:                     # nobody answered
        return None, [], 0

    label = CHOICES[np.argmax(outs[best[0]])]
    if size < req_votes:                 # largest group below the plurality threshold
        return None, best.tolist(), size
    return label, best.tolist(), size


def aggr_baseline(answers,CHOICES):
    """returns (label, indices in the majority component, total_aggr)"""
    outs = [one_hot(a,CHOICES) for a in answers]
    
    n = len(outs)
    
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = similarity(outs[i], outs[j])
    

    _, comp = connected_components(A, directed=False)
    

    best, size = None, 0
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        if outs[idx[0]].sum() == 0:      # skip unanswered singletons
            continue
        if len(idx) > size:
            best, size = idx, len(idx)

    

    if best is None or size < 2:
        return np.nan, [], size if best is not None else 0
    return CHOICES[np.argmax(outs[best[0]])], best.tolist(), size
