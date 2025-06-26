import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig, cossin, det, schur

_validate_default = True


def real_eig(mat):
    """Compute real eigenvectors of a symmetric matrix A.T=A."""
    # n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    new_eigvals = np.zeros_like(eigvals)
    new_eigvecs = np.zeros_like(eigvecs)

    eig_idx = 0
    new_eig_idx = 0
    while eigvecs.shape[1]:
        vec = eigvecs[:, 0]
        _norm = norm(vec)
        if _norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec /= _norm
        alt_vec = vec.conj()
        non_zero_idx = np.where(vec)[0][0]
        conj_phase = alt_vec[non_zero_idx] / vec[non_zero_idx]
        if np.allclose(alt_vec, conj_phase * vec):
            #print(vec)
            #print(f"{conj_phase=}")
            vec *= np.sqrt(conj_phase)
            #print(vec)
            #assert np.allclose(vec.imag, 0.0, atol=3e-8)
            vec = vec.real
            new_eigvecs[:, new_eig_idx] = vec
            new_eigvals[new_eig_idx] = eigvals[eig_idx]
            new_eig_idx += 1
            # Remove used eigvec and project out contribution of remaining eigvecs in directions
            # of vec
            eigvecs = eigvecs[:, 1:]
            eigvecs -= np.outer(vec, vec.conj() @ eigvecs)
        else:
            overlap = np.dot(vec, vec)
            new_vec = (-0.5 * np.angle(overlap)) * vec
            vec1 = new_vec.real / norm(new_vec.real)
            vec2 = new_vec.imag / norm(new_vec.imag)
            new_eigvecs[:, new_eig_idx] = vec1
            new_eigvecs[:, new_eig_idx + 1] = vec2
            new_eigvals[new_eig_idx] = new_eigvals[new_eig_idx + 1] = eigvals[eig_idx]
            new_eig_idx += 2
            # Remove used eigvec and project out contribution of remaining eigvecs in directions
            # of vec1 and vec2
            eigvecs = eigvecs[:, 1:]
            eigvecs -= np.outer(vec1, vec1.conj() @ eigvecs)
            eigvecs -= np.outer(vec2, vec2.conj() @ eigvecs)

        eig_idx += 1

    return new_eigvals, new_eigvecs


def gram_schmidt(vecs):

    # vecs is a 2d array where each column represents a vector;
    # returns g-s orthonormalised 2d array of same shape

    n, m = vecs.shape
    orthogonalised = np.zeros((n, m), dtype=vecs.dtype)

    for i in range(m):

        vec = vecs[:, i]

        for j in range(i):
            proj = np.dot(orthogonalised[:, j].conj(), vec) / (norm(orthogonalised[:, j]) ** 2)
            vec = vec - proj * orthogonalised[:, j]

        _norm = norm(vec)
        if _norm > 1e-14:
            orthogonalised[:, i] = vec / _norm
        else:
            raise ValueError(f"Vector {i} is linearly dependent or nearly zero; cannot normalise.")

    return orthogonalised


def _ai_kak(u, validate=_validate_default):

    # u is a (square) np.array to ai-kak;
    # we follow the procedure in the overleaf

    dim = u.shape[0]
    evals, o1 = eig(u @ u.T)

    # degenerate (irl this means approximately degenerate) eigenvals can produce complex vectors,
    # but we can ``realise'' (lol) them as discussed in the overleaf.
    # note np.linalg.eig does not by default produce orthogonal eigenvecs,
    # and even if it did this wouldn't survive the realising,
    # so we have to press the gram-schmidt button.

    unique_evals = np.unique(evals.round(5))
    if len(unique_evals) != dim:

        for v in unique_evals:

            inds = np.where(np.abs(evals - v) < 1e-5)[0]
            if inds.shape[0] == 1:  # lonely eigenvalue
                continue

            o1[:, inds] = gram_schmidt(o1[:, inds].real)

    if det(o1) < 0:
        o1[:, 0] *= -1

    d = np.diag(np.sqrt(evals))
    o2 = np.conj(d) @ o1.T @ u

    if validate:
        # note somewhat large tolerance values; funny numerical behaviour for n > 75,
        # where n > 75 seems to be a suprisingly precise statement...

        assert np.allclose(o1.imag, 0.0, atol=1e-6)
        assert np.allclose(o1 @ o1.T, np.eye(dim), atol=1e-6)
        assert np.allclose(u @ u.T, o1 @ d @ d @ o1.T, atol=1e-6)
        assert np.allclose(u, o1 @ d @ np.conj(d) @ o1.T @ u, atol=1e-6)
        assert np.allclose(o2.T @ o2, np.eye(dim), atol=1e-6)

    return o1, d, o2


def ai_kak(u, validate=_validate_default):

    Delta = u @ u.T
    evals, o1 = real_eig(u @ u.T)

    if det(o1) < 0:
        o1[:, 0] *= -1

    d = np.diag(np.sqrt(evals))
    o2 = np.conj(d) @ o1.T @ u
    if det(o2) < 0:
        # Instead of guaranteeing the correct determinant while taking the square root,
        # we correct it after the fact
        o2[0] *= -1
        d[0] *= -1

    if validate:
        # note somewhat large tolerance values; funny numerical behaviour for n > 75,
        # where n > 75 seems to be a suprisingly precise statement...

        dim = u.shape[0]
        assert np.allclose(o1.imag, 0.0, atol=1e-6)
        assert np.allclose(o1 @ o1.T, np.eye(dim), atol=1e-6)
        assert np.allclose(u @ u.T, o1 @ d @ d @ o1.T, atol=1e-6)
        assert np.allclose(u, o1 @ d @ np.conj(d) @ o1.T @ u, atol=1e-6)
        assert np.allclose(o2.T @ o2, np.eye(dim), atol=1e-6)

    return o1, d, o2


def sympl_eig(mat, J, subspace):
    """subspace="horizontal" implies that the matrix is skew-symplectic.
    subspace="vertical" implies that the matrix is symplectic."""
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    new_eigvals = np.zeros_like(eigvals)
    new_eigvecs = np.zeros_like(eigvecs)
    if subspace == "horizontal":
        conjugate_eigval = lambda x: x
    else:
        conjugate_eigval = np.conj

    eig_idx = 0
    new_eig_idx = 0
    while eigvecs.shape[1]:
        vec = eigvecs[:, 0]
        _norm = norm(vec)
        if _norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec /= _norm
        alt_vec = J @ vec.conj()
        new_eigvecs[:, new_eig_idx] = alt_vec
        new_eigvecs[:, new_eig_idx + n] = vec
        new_eigvals[new_eig_idx] = conjugate_eigval(eigvals[eig_idx])
        new_eigvals[new_eig_idx + n] = eigvals[eig_idx]
        new_eig_idx += 1

        # Remove used eigvec and project out contribution of remaining eigvecs in directions
        # of vec and alt_vec
        eigvecs = eigvecs[:, 1:]
        eig_idx += 1
        eigvecs -= np.outer(vec, vec.conj() @ eigvecs)
        eigvecs -= np.outer(alt_vec, alt_vec.conj() @ eigvecs)

    return new_eigvals, new_eigvecs


def sympl_real_eig_diii(mat, J):
    assert mat.shape[0] % 2 == 0
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    quadruples = np.zeros_like(eigvecs)
    quadruple_eigvals = np.zeros_like(eigvals)
    minus_one_eigvecs = []
    plus_one_eigvecs = []

    eig_idx = 0
    d = 0
    while eigvecs.shape[1]:
        vec0 = eigvecs[:, 0]
        _norm = norm(vec0)
        if _norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec0 /= _norm
        vec2 = vec0.conj()

        non_zero_idx = np.where(vec0)[0][0]
        conj_phase = vec2[non_zero_idx] / vec0[non_zero_idx]
        if np.allclose(vec2, conj_phase * vec0):
            vec0 *= np.sqrt(conj_phase)
            assert np.allclose(vec0.imag, 0.0)
            vec0 = vec0.real
            vecs = [J @ vec0, vec0]

            # store real symplectic eigenvectors
            assert np.isclose(eigvals[eig_idx], 1.0) or np.isclose(eigvals[eig_idx], -1.0)
            if eigvals[eig_idx] > 0:
                plus_one_eigvecs.extend(vecs)
            else:
                minus_one_eigvecs.extend(vecs)
            remove_vecs = vecs
        else:
            overlap = np.dot(vec0, vec0)
            new_vec = (-0.5 * np.angle(overlap)) * vec0
            vec2 = new_vec.real / norm(new_vec.real)
            vec3 = new_vec.imag / norm(new_vec.imag)
            vec0 = J @ vec2
            vec1 = J @ vec3
            # quad = [J @ vec2, J @ vec3, vec2, vec3]
            quadruples[:, 2 * d] = vec0
            quadruples[:, 2 * d + 1] = vec1
            quadruples[:, 2 * d + n] = vec2
            quadruples[:, 2 * d + n + 1] = vec3
            quadruple_eigvals[2 * d] = quadruple_eigvals[2 * d + n + 1] = eigvals[eig_idx]
            quadruple_eigvals[2 * d + 1] = quadruple_eigvals[2 * d + n] = np.conj(eigvals[eig_idx])

            d += 1
            remove_vecs = [vec0, vec1, vec2, vec3]

        # Remove used eigvec and project out contribution of remaining eigvecs in directions
        # of single_vec
        eigvecs = eigvecs[:, 1:]
        for _vec in remove_vecs:
            eigvecs -= np.outer(_vec, _vec.conj() @ eigvecs)

        eig_idx += 1

    two_f = len(minus_one_eigvecs)
    assert two_f % 4 == 0
    f = two_f // 2

    main_diag = np.real(quadruple_eigvals)
    print(quadruples.shape)
    if f > 0:
        minus_one_eigvecs = np.stack(minus_one_eigvecs)
        quadruples[:, 2 * d : 2 * d + f] = minus_one_eigvecs[::2].T
        quadruples[:, 2 * d + n : 2 * d + f + n] = minus_one_eigvecs[1::2].T
        main_diag[2 * d : 2 * d + f] = -1
        main_diag[2 * d + n : 2 * d + f + n] = -1
    if n - (2 * d + f) > 0:
        plus_one_eigvecs = np.stack(plus_one_eigvecs)
        quadruples[:, 2 * d + f : n] = plus_one_eigvecs[::2].T
        quadruples[:, 2 * d + n + f : 2 * n] = plus_one_eigvecs[1::2].T
        main_diag[2 * d + f : n] = 1
        main_diag[2 * d + f + n : 2 * n] = 1

    upper_diag = np.zeros(2 * n)
    upper_diag = -np.imag(quadruple_eigvals)
    if n % 2:
        upper_diag[1:n:2] = 0
        upper_diag[n + 1 :: 2] = 0
    else:
        upper_diag[1::2] = 0
    upper_diag = upper_diag[: 2 * n - 1]
    mu = np.diag(main_diag) + np.diag(upper_diag, k=1) - np.diag(upper_diag, k=-1)

    return mu, quadruples


def sympl_real_eig_ci(mat, J):
    assert mat.shape[0] % 2 == 0
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    new_eigvecs = np.zeros_like(eigvecs)
    new_eigvals = np.zeros_like(eigvals)
    # minus_one_eigvecs = []
    # plus_one_eigvecs = []

    eig_idx = 0
    new_eig_idx = 0
    # d = 0
    while eigvecs.shape[1]:
        vec0 = eigvecs[:, 0]
        _norm = norm(vec0)
        if _norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec0 /= _norm
        vec2 = vec0.conj()

        non_zero_idx = np.where(vec0)[0][0]
        conj_phase = vec2[non_zero_idx] / vec0[non_zero_idx]
        if np.allclose(vec2, conj_phase * vec0):
            vec0 *= np.sqrt(conj_phase)
            assert np.allclose(vec0.imag, 0.0)
            vec0 = vec0.real
            vec1 = J @ vec0

            # store real symplectic eigenvectors
            new_eigvecs[:, new_eig_idx] = vec1
            new_eigvecs[:, new_eig_idx + n] = vec0
            new_eigvals[new_eig_idx] = np.conj(eigvals[eig_idx])
            new_eigvals[new_eig_idx + n] = eigvals[eig_idx]
            remove_vecs = [vec0, vec1]
            new_eig_idx += 1
        else:
            overlap = np.dot(vec0, vec0)
            new_vec = (-0.5 * np.angle(overlap)) * vec0
            vec2 = new_vec.real / norm(new_vec.real)
            vec3 = new_vec.imag / norm(new_vec.imag)
            vec0 = J @ vec2
            vec1 = J @ vec3
            new_eigvecs[:, new_eig_idx] = vec0
            new_eigvecs[:, new_eig_idx + 1] = vec1
            new_eigvecs[:, new_eig_idx + n] = vec2
            new_eigvecs[:, new_eig_idx + n + 1] = vec3
            new_eigvals[new_eig_idx] = new_eigvals[new_eig_idx + 1] = np.conj(eigvals[eig_idx])
            new_eigvals[new_eig_idx + n] = new_eigvals[new_eig_idx + n + 1] = eigvals[eig_idx]

            new_eig_idx += 2
            remove_vecs = [vec0, vec1, vec2, vec3]

        # Remove used eigvec and project out contribution of remaining eigvecs in directions
        # of single_vec
        eigvecs = eigvecs[:, 1:]
        for _vec in remove_vecs:
            eigvecs -= np.outer(_vec, _vec.conj() @ eigvecs)

        eig_idx += 1

    return new_eigvals, new_eigvecs


def J_n(n):
    eye = np.eye(n)
    z = np.zeros((n, n))
    return np.block([[z, eye], [-eye, z]])


def aii_kak(u, validate=_validate_default):
    dim = u.shape[0]
    assert dim % 2 == 0
    J = J_n(dim // 2)

    Delta = u @ J @ u.T @ J.T
    eigvals, s1 = sympl_eig(Delta, J, "horizontal")
    d = np.diag(np.sqrt(eigvals))
    s2 = np.conj(d) @ s1.conj().T @ u

    if validate:
        assert np.allclose(J @ s1.conj() @ J.T, s1, atol=1e-6)  # s1 is symplectic
        assert np.allclose(J @ d.conj() @ J.T, d.conj(), atol=1e-6)  # s1 is symplectic
        assert np.allclose(s1 @ s1.conj().T, np.eye(dim), atol=1e-6)  # s1 is unitary
        assert np.allclose(Delta, s1 @ d @ d @ s1.conj().T, atol=1e-6)  # s1 is a horizontal CD

        assert np.allclose(s2 @ s2.conj().T, np.eye(dim), atol=1e-6)  # s2 is unitary
        assert np.allclose(
            J @ s2.conj() @ J.T, s2, atol=1e-6
        ), f"\n{J @ s2.conj() @ J.T=}\n{s2=}"  # s2 is symplectic
        assert np.allclose(u, s1 @ d @ s2, atol=1e-6)  # s1 and s2 make a KAK decomp

    return s1, d, s2


def aiii_kak(u, p, q, validate=_validate_default):
    assert u.shape == (p + q, p + q)
    if p == 0 or q == 0:
        return u, np.eye(p + q), np.eye(p + q)
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(u, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)
        f[:, :p] = np.roll(f[:, :p], q - p, axis=1)
        f[:p] = np.roll(f[:p], q - p, axis=0)

    if validate:
        dim = u.shape[0]
        r = min(p, q)
        s = max(p, q)
        assert np.allclose(k1[:p, p:], 0.0) and np.allclose(k1[p:, :p], 0.0)
        assert np.allclose(k1 @ k1.conj().T, np.eye(dim))
        assert np.allclose(k2[:p, p:], 0.0) and np.allclose(k2[p:, :p], 0.0)
        assert np.allclose(k2 @ k2.conj().T, np.eye(dim))

        assert np.allclose(f[r:s, r:s], np.eye(s - r))
        assert np.allclose(np.diag(np.diag(f[:r, :r])), f[:r, :r])
        assert np.allclose(f[:r, :r], f[s:, s:])
        assert np.allclose(np.diag(np.diag(f[:r, s:])), f[:r, s:])
        assert np.allclose(f[:r, s:], -f[s:, :r])
        assert np.allclose(f[r:s, :r], 0.0)
        assert np.allclose(f[r:s, s:], 0.0)
        assert np.allclose(f[:r, r:s], 0.0)
        assert np.allclose(f[s:, r:s], 0.0)
        assert np.allclose(k1 @ f @ k2, u), f"\n{k1}\n{f}\n{k2}\n{k1 @ f @ k2}\n{u}"

    return k1, f, k2


def bdi_kak(o, p, q, validate=_validate_default):
    """BDI(p, q) Cartan decomposition of special orthogonal o

    Args:
        o (np.ndarray): The special orthogonal matrix to decompose. It must be square-shaped with
            size p+q.
        p (int): First subspace size for SO(p) x SO(q), the vertical subspace
        q (int): Second subspace size for SO(p) x SO(q), the vertical subspace

    Returns:
        np.ndarray: The first K from the KAK decomposition
        np.ndarray: The exponentiated Cartan subalgebra element A from the KAK decomposition
        np.ndarray: The second K from the KAK decomposition


    The input ``o`` and all three output matrices are group elements, not algebra elements.
    """
    assert o.shape == (p + q, p + q)
    if p == 0 or q == 0:
        return o, np.eye(p + q), np.eye(p + q)
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(o, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)
        f[:, :p] = np.roll(f[:, :p], q - p, axis=1)
        f[:p] = np.roll(f[:p], q - p, axis=0)

    # banish negative determinants
    d1p, d1q = det(k1[:p, :p]), det(k1[p:, p:])
    d2p, d2q = det(k2[:p, :p]), det(k2[p:, p:])
    assert np.isclose(d1p * d1q * d2p * d2q, 1.0), f"{d1p * d1q * d2p * d2q}"
    s = max(p, q)

    k1[:, 0] *= d1p
    k1[:, s] *= d1q
    k2[0] *= d2p
    k2[s] *= d2q

    f[:, 0] *= d1p
    f[:, s] *= d1q
    f[0] *= d2p
    f[s] *= d2q

    if validate:
        r = min(p, q)
        assert np.allclose(k1[:p, p:], 0.0) and np.allclose(k1[p:, :p], 0.0)
        assert np.allclose(k1 @ k1.conj().T, np.eye(p + q))
        assert np.allclose(k2[:p, p:], 0.0) and np.allclose(k2[p:, :p], 0.0)
        assert np.allclose(k2 @ k2.conj().T, np.eye(p + q))
        assert np.allclose(k1.imag, 0.0) and np.allclose(k2.imag, 0.0)

        assert np.allclose(f[r:s, r:s], np.eye(s - r))
        assert np.allclose(np.diag(np.diag(f[:r, :r])), f[:r, :r])
        assert np.allclose(f[:r, :r], f[s:, s:])
        assert np.allclose(np.diag(np.diag(f[:r, s:])), f[:r, s:])
        assert np.allclose(f[:r, s:], -f[s:, :r])
        assert np.allclose(f[r:s, :r], 0.0)
        assert np.allclose(f[r:s, s:], 0.0)
        assert np.allclose(f[:r, r:s], 0.0)
        assert np.allclose(f[s:, r:s], 0.0)
        assert np.allclose(k1 @ f @ k2, o), f"\n{k1}\n{f}\n{k2}\n{k1 @ f @ k2}\n{o}"
        assert np.allclose(
            [det(k1[:p, :p]), det(k1[p:, p:]), det(k2[:p, :p]), det(k2[p:, p:])], 1.0
        )

    return k1, f, k2


def schur_sqrt(u):
    dim = u.shape[0]
    if dim % 2 == 1:
        idx = np.where(np.isclose(np.diag(u), 1.0))[0][0]
        sliced_u = np.block(
            [[u[:idx, :idx], u[:idx, idx + 1 :]], [u[idx + 1 :, :idx], u[idx + 1 :, idx + 1 :]]]
        )
        sl_sqrt = schur_sqrt(sliced_u)
        sqrt = np.block(
            [
                [sl_sqrt[:idx, :idx], np.zeros((idx, 1)), sl_sqrt[:idx, idx:]],
                [np.zeros((1, idx)), 1.0, np.zeros((1, dim - idx - 1))],
                [sl_sqrt[idx:, :idx], np.zeros((dim - idx - 1, 1)), sl_sqrt[idx:, idx:]],
            ]
        )
        return sqrt
    sqrt = np.copy(u)
    iY = np.array([[0, 1], [-1, 0]])
    for i in range(0, dim - 1, 2):
        if np.isclose(u[i, i + 1], 0.0):
            if u[i, i] < 0:
                sqrt[i : i + 2, i : i + 2] = iY
        else:
            theta = np.arctan2(u[i, i + 1], u[i, i]) / 2
            sqrt[i : i + 2, i : i + 2] = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )

    return sqrt


def diii_kak(o, validate=_validate_default):
    dim = o.shape[0]
    assert dim % 2 == 0
    n = dim // 2
    J = J_n(n)

    Delta = o @ J @ o.T @ J.T
    # A_squared is mu^2\oplus mu^2^T
    A_squared, u1 = sympl_real_eig_diii(Delta, J)

    mu_squared = A_squared[:n, :n]
    print(np.round(A_squared, 3))
    assert np.allclose(mu_squared, A_squared[n:, n:].T)
    assert np.allclose(A_squared[n:, :n], 0.0) and np.allclose(A_squared[:n, n:], 0.0)
    mu = schur_sqrt(mu_squared)
    # A is mu \oplus mu^T
    A = np.block([[mu, np.zeros_like(mu)], [np.zeros_like(mu), mu.T]])
    u2 = A.T @ u1.conj().T @ o

    if validate:
        assert np.allclose(u1.imag, 0.0)  # u1 is real
        assert np.allclose(u1 @ u1.conj().T, np.eye(dim), atol=1e-6)  # u1 is unitary
        print(np.round(u1, 3))
        print(np.round(J @ u1.conj() @ J.T, 3))
        assert np.allclose(J @ u1.conj() @ J.T, u1, atol=1e-6)  # u1 is symplectic
        assert np.allclose(A @ A, A_squared, atol=1e-6)
        assert np.allclose(u1 @ A_squared @ u1.conj().T, Delta, atol=1e-6)  # u1 is a Schur decomp.
        assert np.allclose(J @ A.conj() @ J.T, A.T, atol=1e-6)  # A is skew-symplectic
        assert np.allclose(Delta, u1 @ A @ A @ u1.conj().T, atol=1e-6)  # u1 is a horizontal CD

        assert np.allclose(u2.imag, 0.0)  # u1 is orthogonal
        assert np.allclose(u2 @ u2.conj().T, np.eye(dim), atol=1e-6)  # u2 is unitary
        assert np.allclose(
            J @ u2.conj() @ J.T, u2, atol=1e-6
        ), f"\n{J @ u2.conj() @ J.T=}\n{u2=}"  # u2 is symplectic
        assert np.allclose(o, u1 @ A @ u2, atol=1e-6)  # u1 and u2 make a KAK decomp

    return u1, A, u2


def ci_kak(s, validate=_validate_default):
    dim = s.shape[0]
    assert dim % 2 == 0
    n = dim // 2
    J = J_n(n)

    Delta = s @ s.T
    D_slash_squared, u1 = sympl_real_eig_ci(Delta, J)

    assert np.allclose(D_slash_squared[:n].conj(), D_slash_squared[n:])
    D = np.sqrt(D_slash_squared[:n])
    A = np.diag(np.concatenate([D, D.conj()]))
    u2 = A.conj().T @ u1.conj().T @ s

    if validate:
        assert np.allclose(u1.imag, 0.0)  # u1 is real
        assert np.allclose(u1 @ u1.conj().T, np.eye(dim), atol=1e-6)  # u1 is unitary
        assert np.allclose(J @ u1.conj() @ J.T, u1, atol=1e-6)  # u1 is symplectic
        assert np.allclose(
            u1 @ np.diag(D_slash_squared) @ u1.conj().T, Delta, atol=1e-6
        )  # u1 is an EVD

        assert np.allclose(J @ A.conj() @ J.T, A, atol=1e-6)  # A is symplectic
        assert np.allclose(Delta, u1 @ A @ A @ u1.conj().T, atol=1e-6)  # u1 is a horizontal CD

        assert np.allclose(u2 @ u2.conj().T, np.eye(dim), atol=1e-6)  # u2 is unitary
        assert np.allclose(u2.imag, 0.0)  # u1 is orthogonal
        assert np.allclose(
            J @ u2.conj() @ J.T, u2, atol=1e-6
        ), f"\n{J @ u2.conj() @ J.T=}\n{u2=}"  # u2 is symplectic
        assert np.allclose(s, u1 @ A @ u2, atol=1e-6)  # u1 and u2 make a KAK decomp

    return u1, A, u2


def symplectify(x, J):
    n = x.shape[0] // 2
    new_x = np.zeros_like(x)
    idx = 0
    while x.shape[1]:
        vec0 = x[:, 0]
        _norm = norm(vec0)
        if _norm < 1e-12:
            x = x[:, 1:]
            continue
        vec0 /= _norm
        vec1 = J @ vec0.conj()
        new_x[:, idx] = vec1
        new_x[:, idx + n] = vec0
        x = x[:, 1:]
        x -= np.outer(vec0, vec0.conj() @ x)
        x -= np.outer(vec1, vec1.conj() @ x)
        idx += 1

    return new_x


def cii_kak(s, p, q, validate=_validate_default):
    # assert p >= q
    n = p + q
    Z = lambda a, b: np.zeros((a, b))
    I = np.eye
    J = J_n(n)

    # to be modded
    chi = np.block(
        [
            [I(p), Z(p, p), Z(p, q), Z(p, q)],
            [Z(q, p), Z(q, p), I(q), Z(q, q)],
            [Z(p, p), I(p), Z(p, q), Z(p, q)],
            [Z(q, p), Z(q, p), Z(q, q), I(q)],
        ]
    )

    d = abs(p - q)
    r = min(p, q)
    # to be modded
    if p >= q:
        eta = np.block(
            [
                [I(r), Z(r, r), Z(r, d), Z(r, n)],
                [Z(d, r), Z(d, r), I(d), Z(d, n)],
                [Z(r, r), I(r), Z(r, d), Z(r, n)],
                [Z(n, r), Z(n, r), Z(n, d), I(n)],
            ]
        )
    else:
        eta = np.block(
            [
                [I(n), Z(n, d), Z(n, r), Z(n, r)],
                [Z(r, n), Z(r, d), I(r), Z(r, r)],
                [Z(d, n), I(d), Z(d, r), Z(d, r)],
                [Z(r, n), Z(r, d), Z(r, r), I(r)],
            ]
        )
    # print(f"{chi.shape=}")
    # print(f"{eta.shape=}")
    # print(f"{s.shape=}")
    # print(eta)
    # print(chi)

    sprime = eta.T @ chi.T @ s @ chi @ eta
    u1, f0, u2 = aiii_kak(sprime, 2 * p, 2 * q, validate=False)
    # print(f"{f0=}")
    v1 = chi @ eta @ u1 @ eta.T @ chi.T
    v2 = chi @ eta @ u2 @ eta.T @ chi.T
    fbar = chi @ eta @ f0 @ eta.T @ chi.T
    # print(fbar)
    K_pq = np.diag(np.concatenate([np.ones(p), -np.ones(q), np.ones(p), -np.ones(q)]))
    if validate:
        assert np.allclose(fbar.imag, 0.0)
        assert np.allclose(fbar @ fbar.T, np.eye(2 * n))
        # to be modded
        assert np.allclose(fbar[:r, :r], fbar[r + d : n, r + d : n])
        assert np.allclose(fbar[:r, r + d : n], -fbar[r + d : n, :r])
        assert np.allclose(fbar[n : n + r, n : n + r], fbar[n + r + d : 2 * n, n + r + d : 2 * n])
        assert np.allclose(fbar[n : n + r, n + r + d : 2 * n], -fbar[n + r + d : 2 * n, n : n + r])
        assert np.allclose(K_pq @ v1 @ K_pq, v1)
        assert np.allclose(K_pq @ v2 @ K_pq, v2)
        assert np.allclose(K_pq @ fbar @ K_pq, fbar.T)
        assert np.allclose(v1 @ fbar @ v2, s)

    print(np.round(fbar, 2))
    v1 = symplectify(v1, J)
    v2 = symplectify(v2, J)
    # for i in range(n):
    # cand = J @ v1[:,n+i].conj()
    # idx = np.where(cand)[0][0]
    # phase = v1[idx, i] / cand[idx]
    # print(np.allclose(v1[:, i], cand * phase))
    # v1[:, i] = J @ v1[:,n+i].conj()
    # v2[i] = v2[n+i].conj() @ J.T
    fring = v1.conj().T @ s @ v2.conj().T
    # assert np.allclose(v1.conj().T @ v1, np.eye(2*n))
    print(v1.conj().T @ v1)
    print(np.round(fring, 2))
    # assert np.allclose(v2.conj().T @ v2, np.eye(2*n))

    return v1, fring, v2


def a_kak(u, validate=_validate_default):
    dim = u.shape[0]
    assert dim % 2 == 0
    n = dim // 2
    assert np.allclose(u[:n, n:], 0.0) and np.allclose(u[n:, :n], 0.0)

    delta = u[:n, :n] @ u[n:, n:].conj().T
    D_squared, u1 = eig(delta)
    D = np.sqrt(D_squared)
    u2 = np.diag(D) @ u1.conj().T @ u[n:, n:]
    z = np.zeros_like(u1)
    doubled_u1 = np.block([[u1, z], [z, u1]])
    doubled_u2 = np.block([[u2, z], [z, u2]])
    doubled_D = np.diag(np.concatenate([D, D.conj()]))

    if validate:
        assert np.allclose(u1 @ u1.conj().T, np.eye(n))  # U1 is unitary
        assert np.allclose(u2 @ u2.conj().T, np.eye(n))  # U2 is unitary
        assert np.allclose(u1 @ np.diag(D_squared) @ u1.conj().T, delta)  # horizontal KAK decomp
        assert np.allclose(doubled_u1 @ doubled_D @ doubled_u2, u)  # KAK decomp upper part

    return doubled_u1, doubled_D, doubled_u2


def bd_kak(o, validate=_validate_default):
    dim = o.shape[0]
    assert dim % 2 == 0
    n = dim // 2
    assert np.allclose(o[:n, n:], 0.0) and np.allclose(o[n:, :n], 0.0)

    delta = o[:n, :n] @ o[n:, n:].T
    mu_squared, o1 = schur(delta)
    if det(o1) < 0:
        o1[:, :2] = np.roll(o1[:, :2], shift=1, axis=1)
        mu_squared[:, :2] = np.roll(mu_squared[:, :2], shift=1, axis=1)
        mu_squared[:2] = np.roll(mu_squared[:2], shift=1, axis=0)
        assert det(o1) > 0
    mu = schur_sqrt(mu_squared)
    o2 = mu @ o1.conj().T @ o[n:, n:]
    z = np.zeros_like(o1)
    doubled_o1 = np.block([[o1, z], [z, o1]])
    doubled_o2 = np.block([[o2, z], [z, o2]])
    doubled_mu = np.block([[mu, z], [z, mu.T]])

    if validate:
        assert np.allclose(o1.imag, 0)  # o1 is real
        assert np.allclose(o1 @ o1.T, np.eye(n))  # o1 is unitary
        assert np.allclose(mu @ mu.T, np.eye(n))
        assert np.allclose(mu.imag, 0.0)
        assert np.allclose(o2.imag, 0)  # o2 is real
        assert np.allclose(o2 @ o2.T, np.eye(n))  # U2 is unitary
        assert np.allclose(o1 @ mu_squared @ o1.T, delta)  # horizontal KAK decomp
        assert np.allclose(doubled_o1 @ doubled_mu @ doubled_o2, o)  # KAK decomp upper part

    return doubled_o1, doubled_mu, doubled_o2


def c_kak(s, validate=_validate_default):
    dim = s.shape[0]
    assert dim % 4 == 0
    n = dim // 4
    J = J_n(n)

    m = 2 * n
    assert np.allclose(s[:m, m:], 0.0) and np.allclose(s[m:, :m], 0.0)

    delta = s[:m, :m] @ s[m:, m:].conj().T
    D_slash_squared, s1 = sympl_eig(delta, J, "vertical")
    D = np.sqrt(D_slash_squared[:n])
    D_slash = np.concatenate([D, D.conj()])
    s2 = np.diag(D_slash) @ s1.conj().T @ s[m:, m:]
    z = np.zeros_like(s1)
    doubled_s1 = np.block([[s1, z], [z, s1]])
    doubled_s2 = np.block([[s2, z], [z, s2]])
    doubled_D_slash = np.diag(np.concatenate([D_slash, D_slash.conj()]))

    if validate:
        assert np.allclose(s1 @ s1.conj().T, np.eye(m))  # s1 is unitary
        assert np.allclose(J @ s1.conj() @ J.T, s1)  # s1 is symplectic
        assert np.allclose(s2 @ s2.conj().T, np.eye(m))  # s2 is unitary
        assert np.allclose(J @ s2.conj() @ J.T, s2)  # s2 is symplectic
        assert np.allclose(D_slash_squared[:n], D_slash_squared[n:].conj())

        assert np.allclose(
            s1 @ np.diag(D_slash_squared) @ s1.conj().T, delta
        )  # horizontal KAK decomp
        assert np.allclose(doubled_s1 @ doubled_D_slash @ doubled_s2, s)  # KAK decomp upper part

    return doubled_s1, doubled_D_slash, doubled_s2
