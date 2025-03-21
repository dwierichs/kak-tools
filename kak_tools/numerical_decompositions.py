import numpy as np
from scipy.linalg import eig, cossin, det

_validate_default = True

# def real_eig(mat, J):
# TODO


def gram_schmidt(vecs):

    # vecs is a 2d array where each column represents a vector;
    # returns g-s orthonormalised 2d array of same shape

    n, m = vecs.shape
    orthogonalised = np.zeros((n, m), dtype=vecs.dtype)

    for i in range(m):

        vec = vecs[:, i]

        for j in range(i):
            proj = np.dot(orthogonalised[:, j].conj(), vec) / (
                np.linalg.norm(orthogonalised[:, j]) ** 2
            )
            vec = vec - proj * orthogonalised[:, j]

        norm = np.linalg.norm(vec)
        if norm > 1e-14:
            orthogonalised[:, i] = vec / norm
        else:
            raise ValueError(f"Vector {i} is linearly dependent or nearly zero; cannot normalise.")

    return orthogonalised


def ai_kak(u, validate=_validate_default):

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


def sympl_eig(mat, J):
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    new_eigvals = np.zeros_like(eigvals)
    new_eigvecs = np.zeros_like(eigvecs)

    eig_idx = 0
    new_eig_idx = 0
    while eigvecs.shape[1]:
        vec = eigvecs[:, 0]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec /= norm
        alt_vec = J @ vec.conj()
        new_eigvecs[:, new_eig_idx] = alt_vec
        new_eigvecs[:, new_eig_idx + n] = vec
        new_eigvals[new_eig_idx] = new_eigvals[new_eig_idx + n] = eigvals[eig_idx]
        new_eig_idx += 1

        # Remove used eigvec and project out contribution of remaining eigvecs in directions
        # of vec and alt_vec
        eigvecs = eigvecs[:, 1:]
        eig_idx += 1
        eigvecs -= np.outer(vec, vec.conj() @ eigvecs)
        eigvecs -= np.outer(alt_vec, alt_vec.conj() @ eigvecs)

    return new_eigvals, new_eigvecs


def sympl_real_eig(mat, J):
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    new_eigvals = np.zeros_like(eigvals)
    new_eigvecs = np.zeros_like(eigvecs)

    eig_idx = 0
    new_eig_idx = 0
    while eigvecs.shape[1]:
        vec = eigvecs[:, 0]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            eigvecs = eigvecs[:, 1:]
            eig_idx += 1
            continue
        vec /= norm
        alt_vec = J @ vec.conj()
        new_eigvecs[:, new_eig_idx] = alt_vec
        new_eigvecs[:, new_eig_idx + n] = vec
        new_eigvals[new_eig_idx] = new_eigvals[new_eig_idx + n] = eigvals[eig_idx]
        new_eig_idx += 1

        # Remove used eigvec and project out contribution of remaining eigvecs in directions
        # of vec and alt_vec
        eigvecs = eigvecs[:, 1:]
        eig_idx += 1
        eigvecs -= np.outer(vec, vec.conj() @ eigvecs)
        eigvecs -= np.outer(alt_vec, alt_vec.conj() @ eigvecs)

    return new_eigvals, new_eigvecs


def J_n(n):
    eye = np.eye(n)
    z = np.zeros((n, n))
    return np.block([[z, eye], [-eye, z]])


def aii_kak(u, J=J_n, validate=_validate_default):
    dim = u.shape[0]
    assert dim % 2 == 0

    if callable(J):
        J = J(dim // 2)

    Delta = u @ J @ u.T @ J.T
    eigvals, s1 = sympl_eig(Delta, J)
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
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(u, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)

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
        assert np.allclose(k1 @ f @ k2, u), f"\n{k1}\n{a}\n{k2}\n{k1 @ a @ k2}\n{u}"

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
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(o, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)

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
        assert np.allclose(k1 @ f @ k2, o), f"\n{k1}\n{a}\n{k2}\n{k1 @ a @ k2}\n{o}"
        assert np.allclose(
            [det(k1[:p, :p]), det(k1[p:, p:]), det(k2[:p, :p]), det(k2[p:, p:])], 1.0
        )

    return k1, f, k2


def diii_kak(o, J=J_n, validate=_validate_default):
    dim = o.shape[0]
    assert dim % 2 == 0

    if callable(J):
        J = J(dim // 2)

    Delta = o @ J @ o.T @ J.T
    mu_squared, u1 = sympl_real_eig(Delta, J)
    mu = schur_sqrt(mu_squared)
    u2 = mu.T @ u1.conj().T @ o

    if validate:
        assert np.allclose(u1.imag, 0.0)  # u1 is orthogonal
        assert np.allclose(J @ u1.conj() @ J.T, u1, atol=1e-6)  # u1 is symplectic
        assert np.allclose(J @ mu.conj() @ J.T, mu.T, atol=1e-6)  # mu is skew-symplectic
        assert np.allclose(u1 @ u1.conj().T, np.eye(dim), atol=1e-6)  # u1 is unitary
        assert np.allclose(Delta, u1 @ mu @ mu @ u1.conj().T, atol=1e-6)  # u1 is a horizontal CD

        assert np.allclose(u2.imag, 0.0)  # u1 is orthogonal
        assert np.allclose(u2 @ u2.conj().T, np.eye(dim), atol=1e-6)  # u2 is unitary
        assert np.allclose(
            J @ u2.conj() @ J.T, u2, atol=1e-6
        ), f"\n{J @ u2.conj() @ J.T=}\n{u2=}"  # u2 is symplectic
        assert np.allclose(u, u1 @ mu @ u2, atol=1e-6)  # u1 and u2 make a KAK decomp

    return u1, mu, u2
