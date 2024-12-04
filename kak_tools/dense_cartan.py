"""Dense implementations of Cartan decompositions."""
import numpy as np
from scipy.linalg import cossin, expm, logm, det

def bdi(u, p, q, is_horizontal=True):
    """BDI(p, q) Cartan decomposition of special orthogonal u

    Args:
        u (np.ndarray): The special orthogonal matrix to decompose. It must be square-shaped with
            size p+q.
        p (int): First subspace size for SO(p) x SO(q), the vertical subspace
        q (int): Second subspace size for SO(p) x SO(q), the vertical subspace
        is_horizontal (bool): Whether or not ``u`` is the exponential of a horizontal element

    Returns:
        np.ndarray: The first K from the KAK decomposition
        np.ndarray: The exponentiated Cartan subalgebra element A from the KAK decomposition
        np.ndarray: The second K from the KAK decomposition


    The input ``u`` and all three output matrices are group elements, not algebra elements.
    """
    k1, a, k2 = cossin(u, p=p, q=q)

    if is_horizontal:
        for i in range(p):
            # we want k1 = k2.T, but scipy is unburdened by such concerns
            if not(np.allclose(k1[:p,i], k2[i,:p])):
                d = np.diag([ (-1) ** ((j%p)==i%p) for j in range(p + q) ])
                a = a @ d
                k2 = d @ k2
                # k1 @ a @ k2 was unchanged by the above business

    # banish negative determinants
    d1 = np.diag([det(k1[:p,:p])] + [1] * (p+q-2) + [det(k1[p:,p:])])
    if is_horizontal:
        d2 = d1
    else:
        d2 = np.diag([det(k2[:p,:p])] + [1] * (p+q-2) + [det(k2[p:,p:])])

    k1 = k1 @ d1
    a = d1 @ a @ d2
    k2 = d2 @ k2

    if is_horizontal:
        assert np.allclose(k1, k2.T)
    assert np.allclose(k1 @ a @ k2, u)

    return k1, a, k2

def embed(op, start, end, n):
    mat = np.eye(n, dtype=complex)
    mat[start:end, start:end] = op
    return mat

def recursive_bdi(U, n, num_iter=None):
    if num_iter is None:
        num_iter = int(np.round(np.log2(n))) - 2

    p = q = n // 2
    k1, a, k2 = bdi(U, p, q)
    ops = {-1: [(U, 0, n, None)], 0: [(k1, 0, n, "k"), (a, 0, n, "a"), (k2, 0, n, "k")]}
    for i in range(1, num_iter+1):
        p = q = p // 2
        new_ops = []
        for k, (op, start, end, _type) in enumerate(ops[i-1]):
            if _type == "a":
                # CSA element
                new_ops.append((op, start, end, "a"))
                continue
            width = end - start
            assert width % 2 == 0
            for s, e in [(start, start+width//2), (start+width//2, end)]:
                k1, a, k2 = bdi(op[s:e, s:e], p, q, is_horizontal=False)
                k1, a, k2 = embed(k1, s, e, n), embed(a, s, e, n), embed(k2, s, e, n)
                new_ops.extend([(k1, s, e, "k"), (a, s, e, "a"), (k2, s, e, "k")])
        ops[i] = new_ops
    return ops
