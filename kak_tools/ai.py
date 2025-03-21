import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eig

np.set_printoptions(precision=3, suppress=1, linewidth=500)

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

PAULI_MATRICES = {"I": I, "X": X, "Y": Y, "Z": Z}


def psm(pauli_string):
    result_matrix = np.array([[1.0]])

    for char in pauli_string:
        char = char.upper()
        if char not in PAULI_MATRICES:
            raise ValueError(f"Invalid Pauli label: {char}")
        result_matrix = np.kron(result_matrix, PAULI_MATRICES[char])

    return result_matrix


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


def ai_kak(u, verify=0):

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

    if np.linalg.det(o1) < 0:
        o1[:, 0] *= -1

    d = np.sqrt(evals)
    o2 = np.conj(d)[:, None] * o1.T @ u
    d = np.diag(d)

    if verify:
        # note somewhat large tolerance values; funny numerical behaviour for n > 75,
        # where n > 75 seems to be a suprisingly precise statement...

        assert np.allclose(o1.imag, 0.0, atol=1e-6)
        assert np.allclose(o1 @ o1.T, np.eye(dim), atol=1e-6)
        assert np.allclose(u @ u.T, o1 @ d @ d @ o1.T, atol=1e-6)
        assert np.allclose(u, o1 @ d @ np.conj(d) @ o1.T @ u, atol=1e-6)
        assert np.allclose(o2.T @ o2, np.eye(dim), atol=1e-6), op.T @ op

    return o1, d, o2


if __name__ == "__main__":

    # u = expm(1j*0.32*psm('XIXY') + 1j*psm('XYYZ'))

    n = 5
    u = unitary_group.rvs(n)
    a, b, c = ai_kak(u, verify=1)

    print(a)
    print(b)
    print(c)
