"""Dense implementations of Cartan decompositions."""

import numpy as np
from itertools import combinations
from pennylane.pauli import PauliSentence
from scipy.linalg import cossin, expm, logm, det


def bdi(u, p, q, is_horizontal=True, validate=True, **kwargs):
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
    if kwargs.get("compute_u", True) is False:
        return cossin(u, p=p, q=p, swap_sign=True, separate=True, **kwargs)[1]
    (k11, k12), theta, (k21, k22) = cossin(u, p=p, q=p, swap_sign=True, separate=True)
    if validate:
        f = abs(p - q)
        r = min(p, q)
        k1 = np.block([[k11, np.zeros((p, q))], [np.zeros((q, p)), k12]])
        a = np.block(
            [
                [np.eye(f), np.zeros((f, r)), np.zeros((f, r))],
                [np.zeros((r, f)), np.diag(np.cos(theta)), np.diag(np.sin(theta))],
                [np.zeros((r, f)), np.diag(-np.sin(theta)), np.diag(np.cos(theta))],
            ]
        )
        k2 = np.block([[k21, np.zeros((p, q))], [np.zeros((q, p)), k22]])
        assert np.allclose(k1 @ a @ k2, u), f"\n{k1}\n{a}\n{k2}\n{k1 @ a @ k2}\n{u}"

    if p > q:
        k11 = np.roll(k11, q - p, axis=1)
        k21 = np.roll(k21, q - p, axis=0)

    if validate:
        k1 = np.block([[k11, np.zeros((p, q))], [np.zeros((q, p)), k12]])
        a = np.block(
            [
                [np.diag(np.cos(theta)), np.zeros((r, f)), np.diag(np.sin(theta))],
                [np.zeros((f, r)), np.eye(f), np.zeros((f, r))],
                [np.diag(-np.sin(theta)), np.zeros((r, f)), np.diag(np.cos(theta))],
            ]
        )
        k2 = np.block([[k21, np.zeros((p, q))], [np.zeros((q, p)), k22]])
        assert np.allclose(k1 @ a @ k2, u), f"\n{k1}\n{a}\n{k2}\n{k1 @ a @ k2}\n{u}"

    if is_horizontal:
        for i in range(p):
            # we want k1 = k2.T, but scipy is unburdened by such concerns
            if not (np.allclose(k11[:, i], k21[i])):
                raise ValueError
                d = np.diag([(-1) ** ((j % p) == i) for j in range(p + q)])
                a = a @ d
                k2 = d @ k2
                # k1 @ a @ k2 was unchanged by the above business

    # banish negative determinants
    d11 = det(k11)
    d12 = det(k12)
    k11[:, 0] *= d11
    k12[:, 0] *= d12
    if is_horizontal:
        k21[0] *= d11
        k22[0] *= d12
        theta[0] *= d11 * d12
    else:
        d21 = det(k21)
        k21[0] *= d21
        k22[0] *= d11 * d12 * d21  # d22 must complement to 1
        theta[0] *= d11 * d12
        if d11 * d21 < 0:
            theta[0] += np.pi

    if validate:
        f = abs(p - q)
        r = min(p, q)
        k1 = np.block([[k11, np.zeros((p, q))], [np.zeros((q, p)), k12]])
        a = np.block(
            [
                [np.diag(np.cos(theta)), np.zeros((r, f)), np.diag(np.sin(theta))],
                [np.zeros((f, r)), np.eye(f), np.zeros((f, r))],
                [np.diag(-np.sin(theta)), np.zeros((r, f)), np.diag(np.cos(theta))],
            ]
        )
        k2 = np.block([[k21, np.zeros((p, q))], [np.zeros((q, p)), k22]])
        if is_horizontal:
            assert np.allclose(k1, k2.T)
        assert np.allclose(k1 @ a @ k2, u), f"\n{k1}\n{a}\n{k2}\n{k1 @ a @ k2}\n{u}"

        assert np.allclose([det(k1), det(k1[:p, :p]), det(k1[p:, p:])], 1.0)
        assert np.allclose([det(k2), det(k2[:p, :p]), det(k2[p:, p:])], 1.0)
        assert np.isclose(det(a), 1.0)

    return k11, k12, theta, k21, k22


def embed(op, start, end, n):
    mat = np.eye(n, dtype=op.dtype)
    mat[start:end, start:end] = op
    return mat


def recursive_bdi(U, n, num_iter=None, first_is_horizontal=True, validate=True, return_all=False):
    p = n // 2
    q = n - p
    k11, k12, theta, k21, k22 = bdi(U, p, q, is_horizontal=first_is_horizontal, validate=validate)
    ops = {
        -1: [(U, 0, n, None)],
        0: [
            (k11, 0, p, "k1"),
            (k12, p, n, "k1"),
            (theta, 0, n, "a0"),
            (k21, 0, p, "k2"),
            (k22, p, n, "k2"),
        ],
    }
    current_ops = ops[0]
    _iter = 0

    decomposed_something = True
    while decomposed_something:
        decomposed_something = False
        new_ops = []

        for k, (op, start, end, _type) in enumerate(current_ops):
            _n = end - start
            if _n <= 1:
                continue
            if _type.startswith("a") or _n == 2:
                # CSA element
                new_ops.append((op, start, end, _type))
                if _type == "a0":
                    # Exploit horizontalness
                    break
                continue
            _p = _n // 2
            _q = _n - _p
            # locs = [(0, _p), (_p, _n)]# if _type == "k1" else [(0, _p), (_p, _n)]
            # for s, e in locs:
            k11, k12, theta, k21, k22 = bdi(op, _p, _q, is_horizontal=False, validate=validate)
            new_ops.extend(
                [
                    (k11, start, start + _p, "k1"),
                    (k12, start + _p, end, "k1"),
                    (theta, start, end, "a"),
                    (k21, start, start + _p, "k2"),
                    (k22, start + _p, end, "k2"),
                ]
            )
            decomposed_something = True

        _iter += 1
        if return_all:
            # Exploit horizontalness
            new_ops.extend(
                (
                    ((-op if _type.startswith("a") else op.T), start, end, _type)
                    for op, start, end, _type in new_ops[:-1][::-1]
                )
            )
            ops[_iter] = new_ops
        current_ops = new_ops
        if _iter == num_iter:
            break

    if return_all:
        return ops
    current_ops.extend(
        (
            ((-op if _type.startswith("a") else op.T), start, end, _type)
            for op, start, end, _type in current_ops[:-1][::-1]
        )
    )
    return current_ops


def angles_to_reducible(theta, s, e, mapping, signs):
    p = (e - s) // 2
    q = (e - s) - p
    op = {
        mapping[(s + i, s + p + i)]: th / 2 / signs[(s + i, s + p + i)]
        for i, th in enumerate(theta)
    }
    return PauliSentence(op)


def angles_to_reducible_str(theta, s, e, mapping):
    p = (e - s) // 2
    q = (e - s) - p
    op = {
        (pw_sign := mapping[(s + i, s + p + i)])[0]: th / 2 / pw_sign[1]
        for i, th in enumerate(theta)
    }
    return op


def group_matrix_to_reducible(matrix, start, mapping, signs):
    """Map a (SO(n)) group element composed of commuting Given's rotations
    into commuting Pauli rotations on the reducible representation given by mapping & signs."""
    op = {}
    seen_ids = set()
    for i, j in zip(*np.where(matrix)):
        if i < j:
            assert i not in seen_ids and j not in seen_ids, f"{matrix}"
            m_ii = matrix[i, i]
            m_jj = matrix[j, j]
            assert np.isclose((sign := np.sign(m_ii)), np.sign(m_jj)) or np.allclose(
                [m_ii, m_jj], 0.0
            ), f"{m_ii}, {m_jj}"
            angle = np.arcsin(matrix[i, j])
            assert angle.dtype == np.float64
            if sign < 0:
                angle = np.pi - angle
            op[mapping[(start + i, start + j)]] = angle / 2 / signs[(start + i, start + j)]
            seen_ids |= {i, j}

    return PauliSentence(op)


def group_matrix_to_reducible_str(matrix, start, mapping):
    """Map a (SO(n)) group element composed of commuting Given's rotations
    into commuting Pauli rotations on the reducible representation given by mapping & signs."""
    # op = {}
    # seen_ids = set()
    # print(matrix.shape)
    # print(np.where(matrix))
    assert matrix.shape == (2, 2)
    # for i, j in zip(*np.where(matrix)):
    # if i < j:
    # assert i not in seen_ids and j not in seen_ids, f"{matrix}"
    m_ii = matrix[0, 0]
    # m_jj = matrix[j, j]
    # assert np.isclose((sign := np.sign(m_ii)), np.sign(m_jj)) or np.allclose([m_ii, m_jj], 0.), f"{m_ii}, {m_jj}"
    angle = np.arcsin(matrix[0, 1])
    # assert angle.dtype == np.float64
    if np.sign(m_ii) < 0:
        angle = np.pi - angle
    pw, sign = mapping[(start + 0, start + 1)]
    # op[pw] = angle / 2 / sign
    # seen_ids |= {0, 1}

    # print(op)
    return {pw: angle / 2 / sign}
    # return op


def map_recursive_decomp_to_reducible(
    recursive_decomp, mapping, signs, time=None, tol=1e-8, validate=False
):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """

    decomp = recursive_decomp
    pauli_decomp = []
    if validate:
        from .map_to_irrep import E

        inv_mapping = {val: key for key, val in mapping.items()}
    n = max([k[1] for k in mapping]) + 1
    for mat, s, e, t in decomp:
        if t.startswith("a"):
            ps = angles_to_reducible(mat, s, e, mapping, signs)
        else:
            ps = group_matrix_to_reducible(mat, s, mapping, signs)
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


def round_angles_to_irreducible_mat(theta, start, end, n, tol):
    p = (end - start) // 2
    q = end - start - p
    f = abs(p - q)
    r = min(p, q)
    theta = np.array([th if np.abs(np.sin(th)) > tol else 0 for th in theta])
    a = np.block(
        [
            [np.diag(np.cos(theta)), np.zeros((r, f)), np.diag(np.sin(theta))],
            [np.zeros((f, r)), np.eye(f), np.zeros((f, r))],
            [np.diag(-np.sin(theta)), np.zeros((r, f)), np.diag(np.cos(theta))],
        ]
    )
    return embed(a, start, end, n)


def round_mat_to_irreducible_mat(mat, start, end, n, tol):
    k = np.where((np.abs(mat) > tol) + (np.abs(mat) < (1 - tol)), mat, np.eye(len(mat)))
    return embed(k, start, end, n)


def round_mult_recursive_decomp_str(recursive_decomp, time, n_so, tol=1e-8):

    out = np.eye(n_so)
    for mat, s, e, t in recursive_decomp:
        if t.startswith("a"):
            if t == "a0":
                if time is not None:
                    mat = mat / time
            mat = round_angles_to_irreducible_mat(mat, s, e, n_so, tol)
        else:
            mat = round_mat_to_irreducible_mat(mat, s, e, n_so, tol)

        out @= mat
    return out


def map_recursive_decomp_to_reducible_str(recursive_decomp, mapping, time=None, tol=1e-8):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """
    pauli_decomp = []
    for mat, s, e, t in recursive_decomp:
        if t.startswith("a"):
            ps = angles_to_reducible_str(mat, s, e, mapping)
        else:
            ps = group_matrix_to_reducible_str(mat, s, mapping)
        if t == "a0":
            if time is not None:
                ps = {key: val / time for key, val in ps.items()}
        if tol is not None:
            ps = {key: val for key, val in ps.items() if np.abs(val) >= tol}
        pauli_decomp.extend(((pw, coeff, t) for pw, coeff in ps.items()))

    return pauli_decomp
