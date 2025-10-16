import scipy
import numpy as np

def diii(m):
    """Decompose a horizontal generator of the DIII Cartan decomposition in canonical form."""
    a, Q = scipy.linalg.schur(m)
    assert m.shape[0] % 2 == 0
    assert np.allclose(Q @ a @ Q.T, m)

    # Reorder columns of Q (rows&columns of a) so that the pairs of blocks have opposite sign.
    vec = np.round(np.diag(a, 1)[::2], 10)
    seen = set()
    flips = []
    for i, val in enumerate(vec):
        if np.isclose(val, 0.0, atol=1e-11):
            continue
        if val in seen:
            flips.append(i)
            # Remove the doubly-seen value from `seen` again. This allows us to handle
            # higher-degenerate eigenvalues, i.e. if there are multiple pairs of blocks with
            # the same \mu_i.
            seen.remove(val)
            continue
        if -val in seen:
            continue
        seen.add(val)

    for i in flips:
        Q[:, 2 * i : 2 * i + 2] = Q[:, 2 * i + 1 : 2 * i - 1 : -1]
        a[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = a[
            2 * i + 1 : 2 * i - 1 : -1, 2 * i + 1 : 2 * i - 1 : -1
        ]
    assert np.allclose(Q @ a @ Q.T, m)

    # Reorder columns of Q (rows&columns of a) so that the pairs of blocks are arranged correctly.
    vec = np.round(np.diag(a, 1)[::2], 10)
    ids = np.argsort(vec)
    n = len(ids)
    ids = list(ids[: n // 2 + n % 2]) + list(reversed(ids[n // 2 + n % 2 :]))
    ids = np.array([2 * i + off for i in ids for off in [0, 1]])

    Q = Q[:, ids]
    a = a[ids][:, ids]
    assert np.allclose(Q @ a @ Q.T, m)

    if np.linalg.det(Q) < 0:
        e = m.shape[0] // 4
        assert (m.shape[0] // 2) % 2
        slice_up = slice(2 * e, 2 * e + 2)
        slice_dn = slice(2 * e + 1, 2 * e - 1, -1)
        Q[:, slice_up] = Q[:, slice_dn]
        a[slice_up, slice_up] = a[slice_dn, slice_dn]
        assert np.allclose(Q @ a @ Q.T, m), f"{Q @ a @ Q.T}\n{m}"

    return a, Q
