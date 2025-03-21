import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt
import sys
from time import time
from random import sample
from scipy.linalg import cossin, expm, logm, det

np.set_printoptions(suppress=1, linewidth=1200, threshold=sys.maxsize, precision=3)

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

PAULI_MATRICES = {"I": I, "X": X, "Y": Y, "Z": Z}


def mf(i, n):

    # 0 \leq i < 2n ; returns Pauli string of i'th Majorana (wrt a certain ordering)

    i += 1
    if i <= n:
        return "Z" * (i - 1) + "Y" + "I" * (n - i)

    i -= n
    return "Z" * (i - 1) + "X" + "I" * (n - i)


def explicit_maja_rep(i, j, n):

    # 0 \leq i, j \leq 2n
    # returns the explicit matrix \pm (E_{ij} - E_{ji}) where the \pm depends on i, j

    r = np.zeros((2 * n, 2 * n))
    r[i, j] = 1
    r[j, i] = -1

    if np.allclose(pauli_mult([1, mf(i, n)], [1, mf(j, n)])[0], -1j):
        r *= -1

    return r


def get_str(m, n, tol=1e-10):

    # takes a 2n x 2n matrix m in the maja rep and returns a Pauli string description of the corresponding operator

    assert np.allclose(m, -m.T)

    p = []
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            if abs(m[i, j]) > tol:
                pm = pauli_mult([1, mf(i, n)], [1, mf(j, n)])
                p.append([m[i, j] if np.allclose(pm[0], 1j) else -m[i, j], pm[1]])

    return p


def acom(p, q):

    # p and q pauli strs; returns 0 (1) if the anticommute (commute)

    assert len(p) == len(q)
    c = 0

    for a, b in zip(p, q):

        if (a == "I") or (b == "I") or (a == b):
            continue

        c += 1

    return c % 2


def find_horizontal(h, majoranas, maja_dict, verbose=0):

    # h is a list of pauli strs
    # majoranas is the list of majoranas in the order given by mf()
    # maja_dict is a dict with keys the second order majoranas and values the indices of the
    # (first order) majoranas whose product yields the keys, wrt the order given by mf()

    n = len(h[0])
    if len(h) > n**2:
        print("no horizontal operator for you")
        return None

    m = horizontal_mf_graph(n)
    # nx.draw(m,
    #        with_labels=True,
    #        node_color='lightblue')
    # plt.show()

    vertices = [p for p in h]

    G = nx.Graph()
    G.add_nodes_from(vertices)

    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if acom(h[i], h[j]):
                G.add_edge(h[i], h[j])
    # print(G)

    # nx.draw(G,
    #        with_labels=True,
    #        node_color='lightblue')
    # plt.show()

    GM = iso.GraphMatcher(m, G)
    # print(GM.subgraph_is_isomorphic())

    t = time()
    for mapping in GM.subgraph_isomorphisms_iter():
        if verbose:
            print("iso_time: ", time() - t)

        # okay this next bit is going to get a tiny bit involved
        # what we're trying to do is produce a new_ordering of the majas wrt which the terms in h are horizontal.
        # (at this point in the code we are satisfied that such an ordering exists)

        new_ordering = 2 * n * [""]

        if verbose:
            print("mapping")
            print(mapping)
            print("maja_dict")
            print(maja_dict)

        # there are various constraints we have to satisfy (I am unconvinced that what I am doing is optimal)
        # on our first run through, we will set some elements that are forced by the presence of multiple
        # h elements in the same "row" ("column") of the frustration graph of h
        # (which at this point we know to be isomorphic to a subgraph of the "canonical BDI frustration graph",
        # allowing us to speak of "rows" and "columns")
        # this isomorphism is encoded by the variable called mapping

        for i in range(n):

            if verbose:
                print(i)
            maja_inds = []
            for k in mapping.keys():
                if k[0] == i:
                    maja_inds.append(maja_dict[mapping[k]])
            # print(maja_inds)
            if len(maja_inds) > 1:
                new_ordering[i + n] = majoranas[
                    set.intersection(*[set(lst) for lst in maja_inds]).pop()
                ]

            maja_inds = []
            for k in mapping.keys():
                if k[1] == i:
                    maja_inds.append(maja_dict[mapping[k]])
            # print(maja_inds)
            if len(maja_inds) > 1:
                new_ordering[i] = majoranas[
                    set.intersection(*[set(lst) for lst in maja_inds]).pop()
                ]
            if verbose:
                print(new_ordering)

        # we loop again, this time adding elements that are *forced* by single element rows / columns
        # (a single element row/column fails to force a choice if there is an ambiguity in which of the factors to use)

        if verbose:
            print("single")
        for i in range(n):

            if verbose:
                print(i)
            maja_inds = []
            for k in mapping.keys():
                if k[0] == i:
                    maja_inds.append(maja_dict[mapping[k]])
            # print(maja_inds)
            if len(maja_inds) == 1:
                if majoranas[maja_inds[0][0]] in new_ordering and not (
                    majoranas[maja_inds[0][1]] in new_ordering
                ):
                    new_ordering[i + n] = majoranas[maja_inds[0][1]]
                elif majoranas[maja_inds[0][1]] in new_ordering and not (
                    majoranas[maja_inds[0][0]] in new_ordering
                ):
                    new_ordering[i + n] = majoranas[maja_inds[0][0]]

            maja_inds = []
            for k in mapping.keys():
                if k[1] == i:
                    maja_inds.append(maja_dict[mapping[k]])
            # print(maja_inds)
            if len(maja_inds) == 1:
                if majoranas[maja_inds[0][0]] in new_ordering and not (
                    majoranas[maja_inds[0][1]] in new_ordering
                ):
                    new_ordering[i] = majoranas[maja_inds[0][1]]
                elif majoranas[maja_inds[0][1]] in new_ordering and not (
                    majoranas[maja_inds[0][0]] in new_ordering
                ):
                    new_ordering[i] = majoranas[maja_inds[0][0]]
            if verbose:
                print(new_ordering)

        for _ in range(1):
            if verbose:
                print("pforced")
            # next we make partially constrained choices corresponding to elements in their own row / column
            # once more unto the loop
            for i in range(n):

                if verbose:
                    print(i)
                maja_inds = []
                for k in mapping.keys():
                    if k[0] == i:
                        maja_inds.append(maja_dict[mapping[k]])
                # print(maja_inds)
                if len(maja_inds) == 1:
                    # if (majoranas[maja_inds[0][0]] in new_ordering) and not(majoranas[maja_inds[0][1]] in new_ordering):
                    if not (majoranas[maja_inds[0][1]] in new_ordering):
                        # new_ordering[i] = majoranas[maja_inds[0][0]]
                        new_ordering[i + n] = majoranas[maja_inds[0][1]]

                maja_inds = []
                for k in mapping.keys():
                    if k[1] == i:
                        maja_inds.append(maja_dict[mapping[k]])
                # print(maja_inds)
                if len(maja_inds) == 1:
                    # if not(majoranas[maja_inds[0][0]] in new_ordering) and (majoranas[maja_inds[0][1]] in new_ordering):
                    if not (majoranas[maja_inds[0][0]] in new_ordering):
                        new_ordering[i] = majoranas[maja_inds[0][0]]
                        # new_ordering[i + n] = majoranas[maja_inds[0][1]]
                if verbose:
                    print(new_ordering)

        # at this point we have some free choices, unconstrained by h

        if verbose:
            print("uforced")
        remaining = [m for m in majoranas if not (m in new_ordering)]

        for i in range(len(new_ordering)):
            if new_ordering[i] == "":
                new_ordering[i] = remaining[0]
                remaining.pop(0)
            if verbose:
                print(new_ordering)
            if not (len(remaining)):
                break

        if verbose:
            print("********")
            print("new ordering:")
            print(new_ordering)

        assert is_horizontal(h, gen_maja_pair_decomp_dict(n, new_ordering))
        return new_ordering

    print("no horizontal operator for you")
    return


def horizontal_mf_graph(n):

    G = nx.Graph()

    # Add all vertices on the grid
    vertices = [(x, y) for x in range(n) for y in range(n)]
    G.add_nodes_from(vertices)

    # Add horizontal edges (entire row)
    for y in range(n):
        row_vertices = [(x, y) for x in range(n)]
        G.add_edges_from(
            [(v1, v2) for i, v1 in enumerate(row_vertices) for v2 in row_vertices[i + 1 :]]
        )

    # Add vertical edges (entire column)
    for x in range(n):
        col_vertices = [(x, y) for y in range(n)]
        G.add_edges_from(
            [(v1, v2) for i, v1 in enumerate(col_vertices) for v2 in col_vertices[i + 1 :]]
        )

    return G


def maja_to_full(o, n):

    # maps the maja rep operator o (2n x 2n) to the explicit (2**n x 2**n) operator

    f = np.zeros((2**n, 2**n)).astype(np.complex128)

    for x in get_str(o, n):
        f += x[0] * psm(x[1]) / 2  # annoying factor of 1/2 is annoying

    return f


def bdi(u, p, q):

    # p, q BDI decomp of u; p + q = u.shape[0]

    k1, a, k2 = cossin(u, p=p, q=q)

    for i in range(p):

        # we want k1 = k2.T, but scipy is unburdened by such concerns

        if not (np.allclose(k1[:p, i], k2[i, :p])):
            d = np.diag([1 - 2 * int((j % p) == (i % p)) for j in range(p + q)])
            a = a @ d
            k2 = d @ k2

            # k1 @ a @ k2 was unchanged by the above business

    # banish negative determinants

    if det(k1[:p, :p]) < 0:
        d = np.diag([1 - 2 * int(not (j)) for j in range(p + q)])
        k1 = k1 @ d
        a = d @ a @ d
        k2 = d @ k2

    if det(k1[p:, p:]) < 0:
        d = np.diag([1 - 2 * int(j == (p + q - 1)) for j in range(p + q)])
        k1 = k1 @ d
        a = d @ a @ d
        k2 = d @ k2

    assert np.allclose(k1, k2.T)
    assert np.allclose(k1 @ a @ k2, u)

    return k1, a, k2


def gen_maja_pair_decomp_dict(n, majoranas):

    # O(n^2); probs a cleverer way but this isn't the bottleneck

    d = {}
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            # d[(i,j)] = pauli_mult([1, mf(i, n)], [1, mf(j, n)])[1]
            d[pauli_mult([1, majoranas[i]], [1, majoranas[j]])[1]] = (i, j)

    return d


def is_horizontal(ham, maja_dict):

    n = len(ham[0])

    for h in ham:

        v = maja_dict[h]
        if v[0] >= n or v[1] < n:
            return 0

    return 1


def verify(m, k1, a, k2, verify_sparse=False):

    assert np.allclose(expm(m), k1 @ a @ k2)

    if verify_sparse:

        full_u = np.zeros((2**n, 2**n)).astype(np.complex128)
        for x in get_str(m, n):
            full_u += x[0] * psm(x[1]) / 2
        full_u = expm(1j * full_u)

        k1f = expm(1j * maja_to_full(logm(k1), n))
        af = expm(1j * maja_to_full(logm(a), n))
        k2f = expm(1j * maja_to_full(logm(k2), n))

        assert np.allclose(full_u, k1f @ af @ k2f) or np.allclose(full_u, -k1f @ af @ k2f), (
            logm(k1),
            det(k1[:n, :n]),
            det(k1[n:, n:]),
        )

    return


def gen_hidden_horizontal(majoranas, n, N):

    # n qubits, N terms in the hamiltonian

    # shuffle majoranas
    sm = sample(majoranas, len(majoranas))

    # pick indices that obviously lead to a horizontal generator wrt this basis;
    # will not be obvious in the original basis

    inds = sample([[i, j + n] for i in range(n) for j in range(n)], N)
    ham = [pauli_mult([1, sm[i[0]]], [1, sm[i[1]]])[1] for i in inds]

    return ham


def psm(pauli_string):
    result_matrix = np.array([[1.0]])

    for char in pauli_string:
        char = char.upper()
        if char not in PAULI_MATRICES:
            raise ValueError(f"Invalid Pauli label: {char}")
        result_matrix = np.kron(result_matrix, PAULI_MATRICES[char])

    return result_matrix


def pauli_mult(p, q):

    # p and q are both  [ coefficient (float), pauli string (string) ]
    # return string thats the product of them, in the same representation

    assert len(p[1]) == len(q[1])

    c = p[0] * q[0]
    r = ""

    for p, q in zip(p[1], q[1]):

        if p == "I":
            r += q
            continue

        elif q == "I":
            r += p
            continue

        elif p == q:
            r += "I"
            continue

        elif p == "X":
            if q == "Y":
                r += "Z"
                c *= 1j
                continue

            else:
                r += "Y"
                c *= -1j
                continue

        elif p == "Y":
            if q == "X":
                r += "Z"
                c *= -1j
                continue

            else:
                r += "X"
                c *= 1j
                continue

        elif p == "Z":
            if q == "X":
                r += "Y"
                c *= 1j
                continue

            else:
                r += "X"
                c *= -1j
                continue

    return [c, r]


t = time()

n = 25

# for i in range(2*n):
#    for j in range(i + 1, 2*n):
#        print(i, j, pauli_mult([1, mf(i, n)], [1, mf(j, n)]))
# exit()

majoranas = [mf(i, n) for i in range(2 * n)]
maja_dict = gen_maja_pair_decomp_dict(n, majoranas)

# ham = ['XYI', 'XXI', 'IYX']
# ham = list(set([ pauli_mult([1, majoranas[np.random.randint(0, n)]], [1, majoranas[np.random.randint(0, n)]])[1] for _ in range(n) ]))
# if 'I' * n in ham:
#    ham.remove('I' * n)
# ham = list(maja_dict.keys())[4:7]

ham = gen_hidden_horizontal(majoranas, n, n)

print("Hamiltonian: ", ham)
print("initial majoranas:")
print(majoranas)
print("initially " + (1 - is_horizontal(ham, maja_dict)) * "not " + "horizontal")

majoranas = find_horizontal(ham, majoranas, maja_dict, verbose=0)

if majoranas is not None:
    maja_dict = gen_maja_pair_decomp_dict(n, majoranas)
    print("final majoranas:")
    print(majoranas)
    print("afterwards, " + (1 - is_horizontal(ham, maja_dict)) * "not " + "horizontal")

exit()


# choose generator of target unitary
# atm we specify it in terms of majoranas; would probably be better to specify in terms of pauli strings
# atm we also take care to pick a generator that is horizontal wrt the specific choice of majoranas made in mf();
# would be better to specify the generator and then a choice of majoranas making it horizontal is automatically sought
# I think I can do this, but havent had time yet (additionally, the currently proposed subgraph-isomorphism
# algorithm for this is probably slow, but probably there is an easier way)

# here we specify the generator as
#      coeff, i, j
# ham = [ [-3, 0, n], [1, 1, n], [1, 1, n + 1], [4, 2, n + 2] ]
#    i.e. -3c_0c_n  +  c_1c_n  +  c_1c_{n+1}  +  4c_2c_{n+2}
# m = sum([ x[0] * explicit_maja_rep(x[1],x[2],n) for x in ham ])

# or just get a random horizontal m

m = np.zeros((2 * n, 2 * n))
m[:n, n:] = np.random.random((n, n))
m[n:, :n] = -m[:n, n:].T

u = expm(m)

bdi(u, n, n)
k1, a, k2 = bdi(u, n, n)

assert np.allclose(u, k1 @ a @ k2)

print("target m:")
print(m)
print("corresponding pauli:")
print(get_str(m, n))
print("k1")
print(get_str(logm(k1), n))
print("a")
print(get_str(logm(a), n))
print("k2")
print(get_str(logm(k2), n))

print(f"time: {time()-t:.5f}")
