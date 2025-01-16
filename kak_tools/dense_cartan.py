"""Dense implementations of Cartan decompositions."""

import numpy as np
from itertools import combinations
from pennylane.pauli import PauliSentence
from scipy.linalg import cossin, expm, logm, det


def bdi(u, p, q, is_horizontal=True, validate=True):
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
    if validate:
        assert u.shape == (p + q, p + q)
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, a, k2 = cossin(u, p=p, q=p, swap_sign=True)
    if p > q:
        # For unequal p and q, scipy puts the identity matrix in a at the start of the larger
        # block, while we chose to always have it between the non-identity cos blocks. For p>q
        # this implies that we need to reorder rows/columns a little, which we do with a
        # permutation matrix.
        # This can change the determinants, but we're fixing that below anyways
        r = min(p, q)
        s = p + q - 2 * r
        sigma = np.block(
            [
                [np.zeros((q, p-q)), np.eye(q), np.zeros((q, q))],
                [np.eye(p-q), np.zeros((p-q, q)), np.zeros((p-q, q))],
                [np.zeros((q, p-q)), np.zeros((q,q)), np.eye(q)],
            ]
        )
        k1 = k1 @ sigma.T
        a = sigma @ a @ sigma.T
        k2 = sigma @ k2

    if validate:
        assert np.allclose(k1 @ a @ k2, u), f"\n{k1}\n{a}\n{k2}\n{u}"

    if is_horizontal:
        for i in range(p):
            # we want k1 = k2.T, but scipy is unburdened by such concerns
            if not (np.allclose(k1[:p, i], k2[i, :p])):
                d = np.diag([(-1) ** ((j % p) == i % p) for j in range(p + q)])
                a = a @ d
                k2 = d @ k2
                # k1 @ a @ k2 was unchanged by the above business

    # banish negative determinants
    d1 = np.diag([det(k1[:p, :p])] + [1] * (p - 1) + [det(k1[p:, p:])] + [1] * (q - 1))
    if is_horizontal:
        d2 = d1
    else:
        d2 = np.diag([det(k2[:p, :p])] + [1] * (p - 1) + [det(k2[p:, p:])] + [1] * (q - 1))

    k1 = k1 @ d1
    a = d1 @ a @ d2
    k2 = d2 @ k2

    if validate:
        if is_horizontal:
            assert np.allclose(k1, k2.T)
        assert np.allclose(k1 @ a @ k2, u), f"\n{k1}\n{a}\n{k2}\n{u}"

        assert np.allclose([det(k1), det(k1[:p, :p]), det(k1[p:, p:])], 1.0)
        assert np.allclose([det(k2), det(k2[:p, :p]), det(k2[p:, p:])], 1.0)
        assert np.isclose(det(a), 1.0)
    return k1, a, k2


def embed(op, start, end, n):
    mat = np.eye(n, dtype=op.dtype)
    mat[start:end, start:end] = op
    return mat


def recursive_bdi(U, n, num_iter=None, first_is_horizontal=True, validate=True):
    q = n // 2
    p = n - q
    k1, a, k2 = bdi(U, p, q, is_horizontal=first_is_horizontal, validate=validate)
    ops = {-1: [(U, 0, n, None)], 0: [(k1, 0, n, "k1"), (a, 0, n, "a0"), (k2, 0, n, "k2")]}
    _iter = 0
    decomposed_something = True
    while decomposed_something:
        decomposed_something = False
        new_ops = []
        for k, (op, start, end, _type) in enumerate(ops[_iter]):
            _n = end - start
            if _type.startswith("a") or _n <= 4:
                # CSA element
                new_ops.append((op, start, end, _type))
                if _type == "a0":
                    break
                continue
            _q = _n // 2
            _p = _n - _q
            locs = [(0, _p), (_p, _n)]# if _type == "k1" else [(0, _p), (_p, _n)]
            for s, e in locs:
                __q = (e - s) // 2
                __p = (e - s) - __q
                k1, a, k2 = bdi(op[s:e, s:e], __p, __q, is_horizontal=False, validate=validate)
                new_ops.extend(
                    [
                        (k1, start + s, start + e, "k1"),
                        (a, start + s, start + e, "a"),
                        (k2, start + s, start + e, "k2"),
                    ]
                )
            decomposed_something = True
        new_ops.extend(((op.T, start, end, _type) for op, start, end, _type in new_ops[:-1][::-1]))
        _iter += 1
        ops[_iter] = new_ops
        if _iter == num_iter:
            break

    return ops

"""
def group_matrix_to_reducible(matrix, mapping, signs, invol_type):
    gen = logm(matrix)
    assert np.allclose(np.diag(gen), 0.)
    return map_matrix_to_reducible(gen, mapping, signs, invol_type)
"""


def group_matrix_to_reducible(matrix, start, mapping, signs, invol_type):
    """Map a (SO(n)) group element composed of commuting Given's rotations
    into commuting Pauli rotations on the reducible representation given by mapping & signs."""
    assert invol_type == "BDI"
    op = {}
    seen_ids = set()
    for i, j in zip(*np.where(matrix)):
        if i < j:
            assert i not in seen_ids and j not in seen_ids
            m_ii = matrix[i, i]
            m_jj = matrix[j, j]
            assert np.isclose((sign := np.sign(m_ii)), np.sign(m_jj)) or np.allclose([m_ii, m_jj], 0.), f"{m_ii}, {m_jj}"
            angle = np.arcsin(matrix[i, j])
            assert angle.dtype == np.float64
            if sign < 0:
                angle = np.pi - angle
            op[mapping[(start + i, start + j)]] = angle / 2 / signs[(start + i, start + j)]
            seen_ids |= {i, j}

    return PauliSentence(op)


def map_recursive_decomp_to_reducible(
    recursive_decomp, mapping, signs, invol_type, time=None, tol=1e-8, validate=False
):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """
    assert invol_type == "BDI"

    decomp = recursive_decomp[max(recursive_decomp.keys())]
    pauli_decomp = []
    if validate:
        from .map_to_irrep import E

        inv_mapping = {val: key for key, val in mapping.items()}
    n = max([k[1] for k in mapping]) + 1
    for mat, s, e, t in decomp:
        ps = group_matrix_to_reducible(mat, s, mapping, signs, invol_type)
        if t == "a0":
            if time is not None:
                ps = ps / time
        if tol is not None:
            ps.simplify(tol=tol)
        pauli_decomp.extend(((pw, coeff, t) for pw, coeff in ps.items()))

        # validate to check some properties. Not required for the actual computation
        if validate:
            assert all(pw1.commutes_with(pw2) for pw1, pw2 in combinations(ps.keys(), r=2))
            if not t.startswith("a"):
                assert len(ps) == 2
            if t == "a0":
                if time is not None:
                    ps = ps * time
            rec_mat = np.eye(n)
            for pw, coeff in ps.items():
                i, j = inv_mapping[pw]
                rec_mat = rec_mat @ expm(E((i, j), n, "BDI") * signs[(i, j)] * coeff)
            if not np.allclose(mat, rec_mat):
                print(np.round(mat, 4))
                print(np.round(rec_mat, 4))
                raise ValueError(
                    "The decomposition into Pauli rotations did not correctly reproduce the matrix."
                )

    return pauli_decomp
