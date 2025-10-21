from __future__ import division

from scipy.cluster.hierarchy import from_mlab_linkage
import numpy as np
import logging
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(
    format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
    level=logging.NOTSET
)
HCODE = -1625


from scipy.stats import rankdata
from scipy.spatial.distance import cdist, squareform

def spearman_footrule_matrix(X, ties="average", normalize=False):
    """
    Pairwise Spearman footrule distances between rows of X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_items)
        Each row is a scoring/ordering over the same items.
    ties : {'average','ordinal','min','max','dense'}, default 'average'
        Rank tie-breaking passed to scipy.stats.rankdata.
    normalize : bool, default False
        If True, divide distances by the maximum possible footrule for length m,
        i.e., m^2/2 for even m, (m^2-1)/2 for odd m.

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        Pairwise footrule distance matrix.
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape

    # Rank each row (1..m). rankdata returns floats; that's fine.
    R = np.vstack([rankdata(row, method=ties) for row in X])

    # L1 distance between rank vectors = footrule
    D = cdist(R, R, metric="cityblock")

    if normalize:
        # Maximum possible footrule distance for permutations of length m
        # Even m: m^2/2; odd m: (m^2-1)/2
        max_f = (m * m) / 2.0 if (m % 2 == 0) else (m * m - 1) / 2.0
        if max_f > 0:
            D = D / max_f
    return D



def elinkage(y=[[1, 2, 4], [7, 3, 7], [6, 9, 0]]):
    alpha = 1  # currently unused

    ed = np.asarray(y, dtype=float)

    # Accept 1) square distance matrix, 2) condensed vector, 3) raw observations
    if ed.ndim == 1:
        # condensed vector -> square matrix
        ed = squareform(ed)
    elif ed.ndim == 2 and ed.shape[0] != ed.shape[1]:
        # likely raw observations (n_samples x n_features)
        # compute pairwise distances then squareform
        ed = squareform(pdist(ed, metric='euclidean'))
        #ed = spearman_footrule_matrix(ed, ties="average", normalize=False)

    # Final validation
    if ed.ndim != 2 or ed.shape[0] != ed.shape[1]:
        raise ValueError(f"Expected a square distance matrix; got shape {ed.shape}")

    epsilon = 1e-3
    n = ed.shape[0]
    if ed.shape[0] != ed.shape[1]:
        raise ValueError("Distance matrix must be square")

    clsizes = np.ones(n, dtype=float)      # cluster sizes
    clindex = np.arange(0, n, dtype=float) # cluster indices
    Z = np.zeros((n - 1, 3), dtype=float)  # linkage return value
    nclus = n

    for merges in range(0, n - 2):
        # Work on a copy to choose the next pair to merge
        B = np.array(ed, copy=True)
        # Exclude diagonal from the minimum search
        np.fill_diagonal(B, np.inf)

        # Locate global minimum
        ind = np.argmin(B)
        h, w = ed.shape
        j = ind % w
        i = ind // w
        minim = B[i, j]

        # Warn if there are multiple minima within epsilon
        if np.count_nonzero(np.isclose(B, minim, atol=epsilon)) > 1:
            logging.warning("Warning, the distance matrix contains two minima within epsilon")

        # Update linkage matrix for this merge
        Z[merges, 0] = clindex[i]
        Z[merges, 1] = clindex[j]
        Z[merges, 2] = ed[i, j]

        # Update distances using your formula
        m1 = clsizes[i]
        m2 = clsizes[j]
        m12 = m1 + m2

        for k in range(0, nclus):
            if k != i and k != j:
                m3 = clsizes[k]
                m = m12 + m3
                ed_ik = ed[i, k]
                ed_jk = ed[j, k]
                ed_ij = ed[i, j]
                ed[i, k] = ((m1 + m3) * ed_ik + (m2 + m3) * ed_jk - m3 * ed_ij) / m
                ed[k, i] = ed[i, k]

        # Remove cluster j (merge into i)
        ed = np.delete(np.delete(ed, j, axis=1), j, axis=0)
        clsizes[i] = m12
        clsizes = np.delete(clsizes, j)
        clindex[i] = n + merges
        clindex = np.delete(clindex, j)
        nclus = n - merges - 1

        # Order leaves so that smaller index is first (for MATLAB-style)
        if Z[merges, 0] > Z[merges, 1]:
            Z[merges, 0], Z[merges, 1] = Z[merges, 1], Z[merges, 0]

    # Handle the final merge (remaining two clusters)
    Z[n - 2, :] = [clindex[0], clindex[1], ed[0, 1]]
    if Z[n - 2, 0] > Z[n - 2, 1]:
        Z[n - 2, 0], Z[n - 2, 1] = Z[n - 2, 1], Z[n - 2, 0]

    # MATLAB linkage format is 1-based indexing
    Z[:, 0:2] += 1.0

    return from_mlab_linkage(Z)
