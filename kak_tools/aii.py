import numpy as np 
import sys
from scipy.linalg import eigh_tridiagonal
from scipy.stats import unitary_group, ortho_group
from scipy.linalg import expm, logm, inv
from time import time
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=1200, suppress=1, threshold=sys.maxsize)

# This is all following arxiv:quant-ph/0402051, modulo a few details they 
# don't specify but which seem to be important

I = np.eye(2)
X = np.array([ [0, 1],
               [1, 0]])
Y = np.array([ [0, -1j],
               [1j, 0]])
Z = np.array([ [1, 0],
                [0, -1]])

PAULI_MATRICES = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def psm(pauli_string):
    result_matrix = np.array([[1.0]])
    
    for char in pauli_string:
        char = char.upper()
        if char not in PAULI_MATRICES:
            raise ValueError(f"Invalid Pauli label: {char}")
        result_matrix = np.kron(result_matrix, PAULI_MATRICES[char])
    
    return result_matrix

def direct_sum(v, w):
    
    m, n = v.shape[1], w.shape[1]
    
    vw = np.zeros((v.shape[0] + w.shape[0], m + n)).astype(np.complex128)
    
    vw[:v.shape[0], :m] = v
    vw[v.shape[0]:, m:] = w
    
    return vw

def R(a,b,i,j,l):

    # Construct an ``R'' matrix of Section IV. B of arxiv:quant-ph/0402051
    
    r = np.eye(2*l).astype(np.complex128) 
    n = np.sqrt(np.abs(a[j,i])**2 + np.abs(b[j,i])**2)

    if n:
        r[j,j] = np.conj(a[j,i]) / n
        r[l+j,l+j] = a[j,i] / n

        r[l+j,j] = np.conj(b[j,i]) / n
        r[j,l+j] = -(b[j,i]) / n

    return r

def rs_loop(h,q,i,l):

    # Another little technical subroutine; idk how much there really is to say here lol
    # we're on page 12 of arxiv:quant-ph/0402051 atm
    
    a = h[:l, :l] 
    b = h[:l, l:] 

    for j in range(max(0,i-1),l):
        r = R(a, b, i, j, l)
        h = r @ h @ r.conj().T
        q = r @ q

    rv = np.array([np.sqrt(np.abs(a[j,i]**2)+np.abs(b[j,i])**2) for j in range(l)])
    rv[:i+1] = 0
    rv[i+1] += np.linalg.norm(rv)

    if np.linalg.norm(rv):
        rv /= np.linalg.norm(rv)

    
    S = np.eye(l) - 2*np.einsum('i,j->ij',rv,rv)  # Grover's algorithm vibes
    S = direct_sum(S, S)
    h = S @ h @ S.T
    q = S @ q

    return h, q

def symp_diag(h):

    # h is a np.array; we will return a np.array q that``symplectically diagonalises'' h
    # (in the sense of Section IV. B of our favourite paper)

    q = np.eye(h.shape[0])
    for i in range(h.shape[0]//2 - 1):
        h, q = rs_loop(h, q, i, h.shape[0]//2)

    # h now real tri-diagonal
    
    h = np.real(h)
    val, vec = eigh_tridiagonal(np.diag(h), np.diag(h, k=1))
    
    # shuffle eigenvecs
    sh = np.zeros((2*l, 2*l))
    for i in range(2*l):
        if i<l:
            if np.abs(vec[0,2*i]) < 1e-6:
                sh[2*i+1,i] = 1
            else:
                sh[2*i+0,i] = 1
        else:
            if np.abs(vec[0,1 + 2*(i-l)]) < 1e-6:
                sh[1+2*(i-l),i] = 1
            else:
                sh[0 + 2*(i-l),i] = 1
    
    
    #q = sh.T @ vec.T @ q
    vs = direct_sum((vec@sh)[:l,:l], (vec@sh)[:l,:l])
    q = vs.T @ q
    
    return q


def aii_kak(v, omega, verify=0):

    # We dutifully follow the recipe of Section IV of the paper:

    # Steps 1 & 2: 
    h = -1j/2 * logm(-v @ omega @ v.T @ omega)
    p = expm(1j * h)

    # Step 3:
    w1 = symp_diag(h)

    # Step 4:
    k = p.conj().T @ v
    d = expm(1j * w1 @ h @ w1.conj().T)

    if verify:
        
        assert np.allclose(d, np.diag(d.diagonal()), atol=1e-6), np.amax(d - np.diag(d.diagonal()))
        assert np.allclose(h @ omega, omega @ h.T, atol=1e-6), np.amax(np.abs(h @ omega - omega @ h.T))   # omega skew-sym condition
        assert np.allclose(k.T @ omega @ k, omega, atol=1e-6), np.amax(np.abs(k.T @ omega @ k - omega))   # symplectic condition
        assert np.allclose(w1.T @ omega @ w1, omega, atol=1e-6), np.amax(np.abs(w1.T @ omega @ w1 - omega)) # symplectic condition

        #                       k        a       k   = v
        assert np.allclose(w1.conj().T @ d @ (w1 @ k), v, atol=1e-6), (w1.conj().T @ d @ (w1 @ k), v)

    # Step 5:
    #          k        a       k  
    return w1.conj().T, d, w1 @ k 

if __name__ == '__main__':

    l = 3  # note: we appear to start running into numerical instability at l = 38 (that is, 76 x 76 dimensional matrices)
           # I have previously seen weird behaviour starting at 75 x 75 matrices (e.g. sharp jumps in execution time) and 
           # suspect there is some low-level funny business going on 
    
    # symp bilinear form
    jn = np.zeros((2*l, 2*l))
    jn[:l,l:] = -np.eye(l)
    jn[l:,:l] = np.eye(l)
    
    u = unitary_group.rvs(2*l)
    
    k, a, kp = aii_kak(u, jn, verify=1)

    print(k)
    print(a)
    print(kp)
