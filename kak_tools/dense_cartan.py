"""Dense implementations of Cartan decompositions."""
import numpy as np
from itertools import combinations
from pennylane.pauli import PauliSentence
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
    k1, a, k2 = cossin(u, p=p, q=q, swap_sign=True)

    if is_horizontal:
        for i in range(p):
            # we want k1 = k2.T, but scipy is unburdened by such concerns
            if not(np.allclose(k1[:p,i], k2[i,:p])):
                d = np.diag([ (-1) ** ((j%p)==i%p) for j in range(p + q) ])
                a = a @ d
                k2 = d @ k2
                # k1 @ a @ k2 was unchanged by the above business

    # banish negative determinants
    d1 = np.diag([det(k1[:p,:p])] + [1] * (p - 1) + [det(k1[p:,p:])] + [1] * (q - 1))
    if is_horizontal:
        d2 = d1
    else:
        d2 = np.diag([det(k2[:p,:p])] + [1] * (p - 1) + [det(k2[p:,p:])] + [1] * (q - 1))

    k1 = k1 @ d1
    a = d1 @ a @ d2
    k2 = d2 @ k2

    if is_horizontal:
        assert np.allclose(k1, k2.T)
    assert np.allclose(k1 @ a @ k2, u)

    assert np.allclose([det(k1), det(k1[:p, :p]), det(k1[p:, p:])], 1.)
    assert np.allclose([det(k2), det(k2[:p, :p]), det(k2[p:, p:])], 1.)
    assert np.isclose(det(a), 1.)
    return k1, a, k2

def embed(op, start, end, n):
    mat = np.eye(n, dtype=op.dtype)
    mat[start:end, start:end] = op
    return mat

def recursive_bdi(U, n, num_iter=None):
    if num_iter is None:
        num_iter = int(np.round(np.log2(n))) - 2

    p = q = n // 2
    k1, a, k2 = bdi(U, p, q, is_horizontal=True)
    ops = {-1: [(U, 0, n, None)], 0: [(k1, 0, n, "k"), (a, 0, n, "a0"), (k2, 0, n, "k")]}
    for i in range(1, num_iter+1):
        p = q = p // 2
        new_ops = []
        for k, (op, start, end, _type) in enumerate(ops[i-1]):
            if _type.startswith("a"):
                # CSA element
                new_ops.append((op, start, end, _type))
                continue
            width = end - start
            assert width % 2 == 0
            for s, e in [(start, start+width//2), (start+width//2, end)]:
                k1, a, k2 = bdi(op[s:e, s:e], p, q, is_horizontal=False)
                k1, a, k2 = embed(k1, s, e, n), embed(a, s, e, n), embed(k2, s, e, n)
                new_ops.extend([(k1, s, e, "k"), (a, s, e, "a"), (k2, s, e, "k")])
        ops[i] = new_ops
    return ops

"""
def group_matrix_to_reducible(matrix, mapping, signs, invol_type):
    gen = logm(matrix)
    assert np.allclose(np.diag(gen), 0.)
    return map_matrix_to_reducible(gen, mapping, signs, invol_type)
"""

def group_matrix_to_reducible(matrix, mapping, signs, invol_type):
    """Map a (SO(n)) group element composed of commuting Given's rotations
    into commuting Pauli rotations on the reducible representation given by mapping & signs."""
    assert invol_type=="BDI"
    op = {}
    seen_ids = set()
    for i, j in zip(*np.where(matrix)):
        if i < j:
            assert i not in seen_ids and j not in seen_ids
            assert np.isclose((sign:=np.sign(matrix[i, i])), np.sign(matrix[j, j]))
            angle = np.arcsin(matrix[i, j])
            assert angle.dtype == np.float64
            if sign < 0:
                angle = np.pi - angle
            op[mapping[(i, j)]] = angle / 2 / signs[(i,j)]
            seen_ids |= {i, j}

    return PauliSentence(op)

def map_recursive_decomp_to_reducible(recursive_decomp, mapping, signs, invol_type, time=None, assertions=False):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """
    assert invol_type == "BDI"

    decomp = recursive_decomp[max(recursive_decomp.keys())]
    pauli_decomp = []
    if assertions:
        from .map_to_irrep import E

        inv_mapping = {val: key for key, val in mapping.items()}
    for mat, _, _, t in decomp:
        ps = group_matrix_to_reducible(mat, mapping, signs, invol_type)
        if t == "a0":
            if time is not None:
                ps = ps / time
        ps.simplify()
        pauli_decomp.extend(((pw, coeff, t) for pw, coeff in ps.items()))

        # Assertions to check some properties. Not required for the actual computation
        if assertions:
            assert all(pw1.commutes_with(pw2) for pw1, pw2 in combinations(ps.keys(), r=2))
            if not t.startswith("a"):
                assert len(ps) == 2
            n = len(mat)
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
                raise ValueError("The decomposition into Pauli rotations did not correctly reproduce the matrix.")
            
    return pauli_decomp
