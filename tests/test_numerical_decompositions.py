import numpy as np
import scipy as sp
import pytest
from kak_tools import (
    a_kak,
    ai_kak,
    aii_kak,
    aiii_kak,
    bd_kak,
    bdi_kak,
    diii_kak,
    c_kak,
    ci_kak,
    cii_kak,
)

def J_n(n):
    eye = np.eye(n)
    z = np.zeros((n, n))
    return np.block([[z, eye], [-eye, z]])

def make_unitary(n):
    """Create a Haar random unitary matrix of size n x n."""
    u = sp.stats.unitary_group.rvs(n)
    assert_unitary(u)
    return u

def make_doubled_unitary(n):
    """Create a Haar random doubled unitary matrix of size 2n x 2n, i.e., a matrix
    of the form A⊕ B for A and B Haar random unitaries of size n x n."""
    return np.block([[make_unitary(n), np.zeros((n, n))], [np.zeros((n, n)), make_unitary(n)]])

def make_orthogonal(n):
    """Create a Haar random special orthogonal matrix of size n x n."""
    O = sp.stats.ortho_group.rvs(n)
    # Flip sign of first row if the determinant is -1
    O[0] *= np.linalg.det(O)
    assert_special_orthogonal(O)
    return O

def make_doubled_orthogonal(n):
    """Create a Haar random doubled orthogonal matrix of size 2n x 2n, i.e., a matrix
    of the form A⊕ B for A and B Haar random special orthogonals of size n x n."""
    return np.block([[make_orthogonal(n), np.zeros((n, n))], [np.zeros((n, n)), make_orthogonal(n)]])

def make_symplectic(n):
    """Create a *non-Haar* random unitary symplectic matrix of size 2n x 2n."""
    A = np.random.random(size=(n,n)) + 1j * np.random.random(size=(n,n))
    A = A - A.conj().T
    B = np.random.random(size=(n,n)) + 1j * np.random.random(size=(n,n))
    B = B + B.T
    S = sp.linalg.expm(np.block([[A, B], [-B.conj().T, A.conj()]]))
    assert_symplectic(S)
    return S

def make_doubled_symplectic(n):
    """Create a *non-Haar* random doubled unitary symplectic matrix of size 4n x 4n, i.e., a matrix
    of the form A⊕ B for A and B (non-Haar) random unitary symplectics of size 2n x 2n."""
    return np.block([[make_symplectic(n), np.zeros((2*n, 2*n))], [np.zeros((2*n, 2*n)), make_symplectic(n)]])


def split_into_four(x, split0=None, split1=None):
    if split0 is None:
        split0 = x.shape[0]//2
    if split1 is None:
        split1 = x.shape[1]//2
    x0, x1 = np.split(x, [split0], axis=0)
    x00, x01 = np.split(x0, [split1], axis=1)
    x10, x11 = np.split(x1, [split1], axis=1)
    return x00, x01, x10, x11

def assert_unitary(x):
    assert np.allclose(x @ x.conj().T, np.eye(x.shape[0]))

def assert_special_orthogonal(x):
    assert np.allclose(x.imag, 0.)
    assert np.allclose(x @ x.T, np.eye(x.shape[0]))
    assert np.isclose(np.linalg.det(x), 1.)

def assert_symplectic(x):
    dim = x.shape[0]
    assert dim % 2 == 0
    J = J_n(dim//2)
    assert np.allclose(J @ x.conj() @ J.T, x)

def assert_block_unitary(x, p, q):
    assert x.shape == (p+q, p+q)
    x00, x01, x10, x11 = split_into_four(x, p, p)
    assert np.allclose(x01, 0.) and np.allclose(x10, 0.)
    assert_unitary(x00)
    assert_unitary(x11)
    return x00, x11

def assert_block_orthogonal(x, p, q):
    x00, x11 = assert_block_unitary(x, p, q)
    assert_special_orthogonal(x00)
    assert_special_orthogonal(x11)

def assert_block_symplectic(x, p, q):
    n = p + q
    assert x.shape == (2 * n, 2 * n)

    split = [p, n, p+n]
    x_splits = np.split(x, split, axis=0)
    x_splits = [np.split(sub_x, split, axis=1) for sub_x in x_splits]
    for i, j in [(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)]:
        assert np.allclose(x_splits[i][j], 0.)
    sympl_block0 = np.block([[x_splits[0][0], x_splits[0][2]], [x_splits[2][0], x_splits[2][2]]])
    sympl_block1 = np.block([[x_splits[1][1], x_splits[1][3]], [x_splits[3][1], x_splits[3][3]]])
    assert_symplectic(sympl_block0)
    assert_symplectic(sympl_block1)

def assert_repeat_unitary(x):
    assert x.shape[0] % 2 == 0
    n = x.shape[0] // 2
    x00, x01, x10, x11 = split_into_four(x)
    assert np.allclose(x01, 0.) and np.allclose(x10, 0.)
    assert np.allclose(x00, x11)
    assert_unitary(x00)
    return x00, x01, x10, x11

def assert_repeat_special_orthogonal(x):
    x00, *_, x11 = assert_repeat_unitary(x)
    assert_special_orthogonal(x00)
    assert_special_orthogonal(x11)

def assert_repeat_symplectic(x):
    x00, *_, x11 = assert_repeat_unitary(x)
    assert_symplectic(x00)
    assert_symplectic(x11)

def assert_diagonal(x):
    assert np.allclose(x, np.diag(np.diag(x)))

def assert_unitary_diagonal(x):
    assert_unitary(x)
    assert_diagonal(x)

def assert_embedded_unitary(x):
    assert_special_orthogonal(x)
    assert_symplectic(x)
        

def assert_schur(x):
    if x.shape[0] % 2:
        # In odd dimensions we slice out a 1x1 block for simplicity.
        idx = np.where(np.isclose(np.diag(x), 1.))[0][0]
        zeros = [x[idx,:idx], x[idx, idx+1:], x[:idx, idx], x[idx+1:, idx]]
        assert all(np.allclose(z, 0.) for z in zeros)
        sliced_x = np.block([[x[:idx, :idx], x[:idx, idx+1:]],[x[idx+1:, :idx], x[idx+1:, idx+1:]]])
        return assert_schur(sliced_x)

    assert_special_orthogonal(x)
    D = np.diag(x)
    C0 = D[::2]
    C1 = D[1::2]
    S0 = np.diag(x, k=1)
    S1 = np.diag(x, k=-1)
    assert np.allclose(C0, C1)
    assert np.allclose(S0, -S1)
    assert np.allclose(C0**2 + S0[::2]**2, 1.)
    rest = x - np.diag(D) - np.diag(S0, k=1) - np.diag(S1, k=-1)
    assert np.allclose(rest, 0.)

def assert_repeat_diagonal(x):
    x00, *_ = assert_repeat_unitary(x)
    assert_unitary_diagonal(x00)

def assert_skew_repeat_diagonal(x):
    x[x.shape[0]//2:] = x[x.shape[0]//2:].conj()
    x00, *_ = assert_repeat_unitary(x)
    assert_unitary_diagonal(x00)
    return x00

def assert_skew_repeat_schur(x):
    n = x.shape[0]//2
    x[n:, n:] = x[n:, n:].T
    x00, *_, x11 = assert_repeat_unitary(x)
    assert_schur(x00)

def assert_sympl_diagonal(x):
    assert_unitary_diagonal(x)
    assert_symplectic(x)

def assert_skew_repeat_sympl_diagonal(x):
    x00 = assert_skew_repeat_diagonal(x)
    assert_symplectic(x00)

def assert_cossin(x, p, q):
    assert_special_orthogonal(x)
    r = min(p, q)
    s = max(p, q)
    C0 = x[:r, :r]
    C1 = x[s:, s:]
    S0 = x[:r, s:]
    S1 = x[s:, :r]
    assert_diagonal(C0)
    assert_diagonal(S0)
    assert np.allclose(C0, C1)
    assert np.allclose(S0, -S1)
    assert np.allclose(np.diag(C0)**2 + np.diag(S0)**2, 1.)
    eye = x[r:s, r:s]
    assert np.allclose(eye, np.eye(s-r)), f"\n{eye}"
    zeros = [x[r:s, :r], x[r:s, s:], x[:r, r:s], x[s:, r:s]]
    assert all(np.allclose(z, 0.) for z in zeros)

def assert_sympl_cossin(x, p, q):
    assert_symplectic(x)
    n = p + q
    assert_cossin(x[:n, :n], p, q)
    assert_cossin(x[n:, n:], p, q)


@pytest.mark.parametrize("n", [2, 3, 4, 7, 10, 25, 40])
@pytest.mark.parametrize("validate", [True, False])
class TestNumericalDecompositions:

    def test_a_kak(self, n, validate):
        """Test the numerical KAK decomposition of type A."""
        u = make_doubled_unitary(n)
        u1, d, u2 = a_kak(u, validate=validate)
        assert np.allclose(u1 @ d @ u2, u)
        assert_repeat_unitary(u1)
        assert_skew_repeat_diagonal(d)
        assert_repeat_unitary(u2)

    def test_ai_kak(self, n, validate):
        """Test the numerical KAK decomposition of type AI."""
        u = make_unitary(n)
        o1, d, o2 = ai_kak(u, validate=validate)
        assert np.allclose(o1 @ d @ o2, u)
        assert_special_orthogonal(o1)
        assert_unitary_diagonal(d)
        assert_special_orthogonal(o2)

    def test_aii_kak(self, n, validate):
        """Test the numerical KAK decomposition of type AII."""
        u = make_unitary(2 * n)
        s1, d, s2 = aii_kak(u, validate=validate)
        assert np.allclose(s1 @ d @ s2, u)
        assert_symplectic(s1)
        assert_repeat_diagonal(d)
        assert_symplectic(s2)

    @pytest.mark.parametrize("p_frac", [0.5, 0.2, 0.1])
    @pytest.mark.parametrize("switch_p_and_q", [False, True])
    def test_aiii_kak(self, n, validate, p_frac, switch_p_and_q):
        """Test the numerical KAK decomposition of type AIII."""
        p = int(p_frac * n)
        q = n - p
        if switch_p_and_q:
            p, q = q, p
        u = make_unitary(n)
        k1, f, k2 = aiii_kak(u, p=p, q=q, validate=validate)
        assert np.allclose(k1 @ f @ k2, u), f"\n{k1 @ f @ k2}\n{u}"
        assert_block_unitary(k1, p=p, q=q)
        assert_cossin(f, p, q)
        assert_block_unitary(k2, p=p, q=q)

    def test_bd_kak(self, n, validate):
        """Test the numerical KAK decomposition of type BD."""
        o = make_doubled_orthogonal(n)
        o1, mu, o2 = bd_kak(o, validate=validate)
        assert np.allclose(o1 @ mu @ o2, o)
        assert_repeat_special_orthogonal(o1)
        assert_skew_repeat_schur(mu)
        assert_repeat_special_orthogonal(o2)

    @pytest.mark.parametrize("p_frac", [0.5, 0.2, 0.1])
    @pytest.mark.parametrize("switch_p_and_q", [False, True])
    def test_bdi_kak(self, n, validate, p_frac, switch_p_and_q):
        """Test the numerical KAK decomposition of type BDI."""
        p = int(p_frac * n)
        q = n - p
        if switch_p_and_q:
            p, q = q, p
        o = make_orthogonal(n)
        k1, f, k2 = bdi_kak(o, p=p, q=q, validate=validate)
        assert np.allclose(k1 @ f @ k2, o), f"\n{k1 @ f @ k2}\n{o}"
        assert_block_orthogonal(k1, p=p, q=q)
        assert_cossin(f, p, q)
        assert_block_orthogonal(k2, p=p, q=q)

    def test_diii_kak(self, n, validate):
        """Test the numerical KAK decomposition of type DIII."""
        o = make_orthogonal(2 * n)
        u1, mu, u2 = diii_kak(o, validate=validate)
        assert np.allclose(u1 @ mu @ u2, o)
        assert_embedded_unitary(u1)
        assert_skew_repeat_schur(mu)
        assert_embedded_unitary(u2)


    def test_c_kak(self, n, validate):
        """Test the numerical KAK decomposition of type C."""
        s = make_doubled_symplectic(n)
        s1, d, s2 = c_kak(s, validate=validate)
        assert np.allclose(s1 @ d @ s2, s)
        assert_repeat_symplectic(s1)
        assert_skew_repeat_sympl_diagonal(d)
        assert_repeat_symplectic(s2)

    def test_ci_kak(self, n, validate):
        """Test the numerical KAK decomposition of type CI."""
        s = make_symplectic(n)
        u1, d, u2 = ci_kak(s, validate=validate)
        assert np.allclose(u1 @ d @ u2, s)
        assert_embedded_unitary(u1)
        assert_sympl_diagonal(d)
        assert_embedded_unitary(u2)

    @pytest.mark.parametrize("p_frac", [0.5, 0.2, 0.1])
    @pytest.mark.parametrize("switch_p_and_q", [False, True])
    def test_cii_kak(self, n, validate, p_frac, switch_p_and_q):
        """Test the numerical KAK decomposition of type CII"""
        p = int(p_frac * n)
        q = n - p
        if switch_p_and_q:
            p, q = q, p
        s = make_symplectic(n)
        k1, f, k2 = cii_kak(s, p=p, q=q, validate=validate)
        #print(np.round(k1, 4))
        #print(np.round(f, 4))
        #print(np.round(k2, 4))
        assert_block_symplectic(k1, p=p, q=q)
        assert_sympl_cossin(f, p, q)
        assert_block_symplectic(k2, p=p, q=q)
        assert np.allclose(k1 @ f @ k2, s), f"\n{np.round(k1 @ f @ k2,3)}\n{np.round(s,3)}"

