from itertools import combinations, product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane.pauli import PauliWord, PauliSentence
from pennylane import X, Y, Z
from .pauli_dlas import anticom_graph_pauli


def _anticom_graph_bdi(n, invol_kwargs):
    """Build the anticommutativity graph of our standard basis of so(n) to be used with
    BDI involutions. If p and q are not n/2 in the involution, they should be provided
    in the ``invol_kwargs`` for the construction of the horizontal AC graph."""
    assert set(invol_kwargs).issubset({"p", "q"})
    p = invol_kwargs.get("p", n // 2)
    q = invol_kwargs.get("q", n - p)
    assert p + q == n, "If p and q are provided, they have to add up to n."

    # edges = [((i, j), (i, l)) for i in range(n) for j in range(i+1, n) for l in range(j+1, n)]
    # edges += [((i, j), (k, i)) for i in range(n) for j in range(i+1, n) for k in range(i)]
    # edges += [((i, j), (j, l)) for i in range(n) for j in range(i+1, n) for l in range(j+1, n)]
    # edges += [((i, j), (k, j)) for i in range(n) for j in range(i+1, n) for k in range(j)]
    # edges = {(i, j): ([(i, l) for l in range(j+1, n)] + [(k, i) for k in range(i)] + [(j, l) for l in range(j+1, n)] + [(k, j) for k in range(j)]) for i, j in combinations(range(n),r=2)}
    # graph = nx.Graph(edges)
    edges_hor = [((i, j), (i, l)) for i in range(p) for j in range(p, n) for l in range(j + 1, n)]
    edges_hor += [((i, j), (k, j)) for i in range(p) for j in range(p, n) for k in range(i + 1, p)]
    horizontal_graph = nx.Graph(edges_hor)
    return horizontal_graph


def _anticom_graph_diii(n, invol_kwargs):
    assert n % 2 == 0
    assert set(invol_kwargs) == set()
    m = n // 2
    # See encoding description somewhere else
    nodes = [
        (i, j, _type, sign)
        for _type in "AB"
        for sign in "+-"
        for i, j in combinations(range(m), r=2)
    ]
    nodes_hor = [(i, j, _type, "-") for _type in "AB" for i, j in combinations(range(m), r=2)]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(n1, n2) for n1, n2 in combinations(graph.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph.add_edges_from(edges)

    graph_hor = nx.Graph()
    graph_hor.add_nodes_from(nodes_hor)
    edges_hor = [
        (n1, n2) for n1, n2 in combinations(graph_hor.nodes(), r=2) if n1[0] in n2 or n1[1] in n2
    ]
    graph_hor.add_edges_from(edges_hor)
    return graph, graph_hor


def _anticom_graph_aiii(n, invol_kwargs):
    assert set(invol_kwargs).issubset({"p", "q"})
    if n % 2 or invol_kwargs.get("p", None) != invol_kwargs.get("q", None):
        raise NotImplementedError("BDI currently only is supported with p=q")
    m = n // 2
    nodes = [(i, j, t) for t in "XYZ" for i, j in combinations(range(m), r=2) if t != Z or i == j]
    nodes_hor = [(i, j, t) for t in "XY" for i, j in product(range(m), range(m, n))]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(n1, n2) for n1, n2 in combinations(graph.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph.add_edges_from(edges)

    graph_hor = nx.Graph()
    graph_hor.add_nodes_from(nodes_hor)
    edges_hor = [
        (n1, n2) for n1, n2 in combinations(graph_hor.nodes(), r=2) if n1[0] in n2 or n1[1] in n2
    ]
    graph_hor.add_edges_from(edges_hor)
    return graph, graph_hor


def anticom_graph_irrep(n, invol_type=None, invol_kwargs=None):
    """Create an anticommutation graph for an irrep of a simple algebra,
    in a basis adapted to a given involution type."""
    # assume even n and BDI* involution for now.
    if invol_type == "BDI":
        return _anticom_graph_bdi(n, invol_kwargs)

    elif invol_type == "DIII":
        return _anticom_graph_diii(n, invol_kwargs)

    elif invol_type == "AIII":
        return _anticom_graph_aiii(n, invol_kwargs)

    NotImplementedError("Only BDI, DIII and AIII are implemented.")


def map_horizontal_subgraph(pauli_graph, horizontal_graph):
    """Initiate a mapping between irrep elements and Pauli words by identifying
    a subgraph in the horizontal anticommutation graph of the former that is isomorphic
    to the anticommutation graph of the latter."""
    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(horizontal_graph, pauli_graph)
    mapping = next(graph_matcher.subgraph_isomorphisms_iter())
    return mapping


def _node_commutator(node1, node2, invol_type):
    if invol_type == "BDI":
        a, b, c, d = sorted(node1 + node2)
        if a == b:
            return (c, d)
        if b == c:
            return (a, d)
        return (a, b)
    elif invol_type == "DIII":
        i, j, t1, s1 = node1
        k, l, t2, s2 = node2
        new_t = "A" if t1 == t2 else "B"
        new_s = "+" if s1 == s2 else "-"
        a, b = {i, j, k, l}.difference({i, j} & {k, l})
        if a > b:
            a, b = b, a
        return (a, b, new_t, new_s)

    raise ValueError


def _node_commutator_sign_bdi(node1, node2):
    if node1[0] == node2[0]:
        if node1[1] < node2[1]:
            return -1
        return 1
    if node1[1] == node2[1]:
        if node1[0] < node2[0]:
            return -1
        return 1
    if node1[0] == node2[1]:
        if node1[1] < node2[0]:
            return 1
        return -1
    if node1[1] == node2[0]:
        if node1[0] < node2[1]:
            return 1
        return -1
    raise ValueError


"""
def _node_commutator_sign_diii(node1, node2):
    i, j, t1, s1 = node1
    k, l, t2, s2 = node2
    if t1 == t2 == "A":
        # Commutator sign does not depend on s1, s2
        if (i == k and j < l) or (j == l and i < k) or (i == l and j > k) or (j == k and i > l):
            return -1
        return 1
    if t1 == t2 == "B":
        if i == k:
            return (-1) ** ((j > l) + (s1 == s2))
        if i == l:
            return (-1) ** ((j > k) + (s1 == "+"))
        if j == k:
            return (-1) ** ((i > l) + (s2 == "+"))
        if j == l:
            return (-1) ** ((i > k) + 1)
    if t1 == "A" and t2 == "B":
        if i == k:
            return -1
        if i == l:
            return (-1) ** (s2 == "+")
        if j == k:
            return 1
        if j == l:
            return (-1) ** (s2 == "-")
    if t1 == "B" and t2 == "A":
        return -1 * _node_commutator_sign_diii(node2, node1)
    raise ValueError
"""


def _node_commutator_sign(node1, node2, invol_type):
    if invol_type == "BDI":
        return _node_commutator_sign_bdi(node1, node2)
    elif invol_type == "DIII":
        return _node_commutator_sign_diii(node1, node2)

    raise ValueError


def map_hor_com_hor(mapping, pauli_graph, invol_type=None):
    """Extend an initialized mapping from a horizontal subspace to horizontal Pauli
    words by computing all accessible first-order commutators."""
    inv_mapping = {val: key for key, val in mapping.items()}
    for p1, p2 in pauli_graph.edges():
        com_node = _node_commutator(inv_mapping[p1], inv_mapping[p2], invol_type)
        mapping[com_node] = p1._commutator(p2)[0]

    return mapping


def _yield_sorted_ids_no_collision(i, j, n):
    # Iterate from 0 to n-1 and return sorted versions of (i, k), (k, j)
    for k in range(i):
        yield (k, i), (k, j)
    for k in range(i + 1, j):
        yield (i, k), (k, j)
    for k in range(j + 1, n):
        yield (i, k), (j, k)


def all_pre_commutators(node, n, invol_type):
    if invol_type == "BDI":
        yield from _yield_sorted_ids_no_collision(*node, n)
    if invol_type == "DIII":
        i, j, t, s = node
        signs = [("+", "+"), ("-", "-")] if s == "+" else [("+", "-"), ("-", "+")]
        types = [("A", "A"), ("B", "B")] if t == "A" else [("A", "B")]
        for ids1, ids2 in _yield_sorted_ids_no_collision(i, j, n // 2):
            for s1, s2 in signs:
                for t1, t2 in types:
                    yield ids1 + (t1, s1), ids2 + (t2, s2)


def choose_generic_first_missing(missing, n, invol_type):
    if invol_type == "BDI":
        cand = missing[0]
        assert cand[0] == 0
        return cand, 0

    if invol_type == "DIII":
        cand = missing[0]
        assert cand[0] == 0
        assert cand[2] == "A"
        assert cand[3] == "+"
        return cand, 0


def anticommuting_nodes(node, n, invol_type):
    if invol_type == "BDI":
        for i in range(1, n):
            yield (0, i)
    if invol_type == "DIII":
        for i in range(1, n):
            yield (0, i, "A", "+")
            yield (0, i, "A", "-")
            yield (0, i, "B", "+")
            yield (0, i, "B", "-")


def map_coms(mapping, missing, missing_ops, n, invol_type):
    """Extend a partial mapping by filling in gaps that are commutators of elements
    that exist already in the mapping."""

    missing_idx = 0
    last_reset_length = -1
    while missing:
        if missing_idx == len(missing):
            # Reset position, starting a new recursion loop
            if len(missing) == last_reset_length:
                # Already reset with missing_idx pointing to the end. No extension seems possible
                return mapping, missing, missing_ops
            last_reset_length = len(missing)
            missing_idx = 0

        node = missing[missing_idx]
        for n1, n2 in all_pre_commutators(node, n, invol_type):
            if n1 in mapping and n2 in mapping:
                com_pw = mapping[n1]._commutator(mapping[n2])[0]
                mapping[node] = com_pw
                missing_ops.remove(com_pw)
                missing.pop(missing_idx)
                break
        else:
            missing_idx += 1

    return mapping, [], []


def map_choice(mapping, missing, missing_ops, n, invol_type):
    node, node_idx = choose_generic_first_missing(missing, n, invol_type)
    ac_nodes = anticommuting_nodes(node, n, invol_type)
    for op_idx, op in enumerate(missing_ops):
        if any(op.commutes_with(mapping[ac_node]) for ac_node in ac_nodes if ac_node in mapping):
            continue
        mapping[node] = op
        missing.pop(node_idx)
        missing_ops.pop(op_idx)
        return mapping, missing, missing_ops
    raise ValueError("No compatible choice could be made from the missing operators.")


sign_map = {
    "X": {"X": 1.0, "Y": -1j, "Z": 1j},
    "Y": {"X": 1j, "Y": 1.0, "Z": -1j},
    "Z": {"X": -1j, "Y": 1j, "Z": 1.0},
}


def _comm_sign_pws(a, b):
    wires = set(a) & set(b)
    return np.prod([sign_map[a[w]][b[w]] for w in wires])


def make_signs(mapping, n, invol_type):
    assert invol_type == "BDI"
    assert len(mapping) == (n**2 - n) // 2
    signs = {(0, i): 1 for i in range(1, n)}
    signs[(0, n // 2)] = -1
    while len(signs) < len(mapping):
        for (n1, pre_sign1), (n2, pre_sign2) in combinations(signs.items(), r=2):
            if not (n1[0] in n2 or n1[1] in n2):
                continue
            node_com_sign = _node_commutator_sign(n1, n2, invol_type)
            com_node = _node_commutator(n1, n2, invol_type)
            pw_com_sign = (_comm_sign_pws(mapping[n1], mapping[n2]) / 1j).real
            # pw_com_sign = (mapping[n1]._commutator(mapping[n2])[1] / -2j).real
            relative_sign = node_com_sign * pre_sign1 * pre_sign2 / pw_com_sign
            if signs.get(com_node, relative_sign) != relative_sign:
                raise ValueError("Inconsistency")
            signs[com_node] = relative_sign
    return signs


def map_simple_to_irrep(ops, horizontal_ops=None, n=None, invol_type=None, invol_kwargs=None):
    """Map a list of Pauli words that is guaranteed to form a simple Lie algebra of type
    ``dla_type`` to the elements in an irreducible representation of the algebra."""
    assert all(isinstance(op, PauliWord) for op in ops)
    assert isinstance(n, int)
    assert invol_type in {"AI", "AII", "AIII", "BDI", "CI", "CII", "DIII"}
    assert invol_type == "BDI"
    if invol_type in {"AII", "DIII"}:
        assert n % 2 == 0
    if invol_kwargs is None:
        invol_kwargs = {}

    if horizontal_ops is None:
        raise NotImplementedError("This is the simpler scenario, but it is not implemented yet.")
    if isinstance(horizontal_ops, dict):
        # If a dictionary is passed, assume that it already contains a mapping for the horizontal
        # operators, rather than just the operators.
        mapping = horizontal_ops
        pauli_graph = anticom_graph_pauli(mapping.values())
        assert all(
            isinstance(key, tuple) and len(key) == 2 and isinstance(op, PauliWord)
            for key, op in horizontal_ops.items()
        )
    else:
        assert all(isinstance(op, PauliWord) for op in horizontal_ops)

        horizontal_graph = anticom_graph_irrep(n, invol_type, invol_kwargs)
        pauli_graph = anticom_graph_pauli(horizontal_ops)

        mapping = map_horizontal_subgraph(pauli_graph, horizontal_graph)

    # This is BDI specific for now.
    all_nodes = list(combinations(range(n), r=2))

    mapping = map_hor_com_hor(mapping, pauli_graph, invol_type=invol_type)
    missing = [node for node in all_nodes if node not in mapping]
    missing_ops = list(set(ops).difference(set(mapping.values())))

    prog_state = (mapping, missing, missing_ops)

    # First completion round
    mapping, missing, missing_ops = map_coms(
        mapping, missing, missing_ops, n, invol_type=invol_type
    )

    while missing:
        assert missing_ops
        mapping, missing, missing_ops = map_choice(mapping, missing, missing_ops, n, invol_type)
        mapping, missing, missing_ops = map_coms(
            mapping, missing, missing_ops, n, invol_type=invol_type
        )

    assert not missing_ops
    assert len(mapping) == len(all_nodes)

    signs = make_signs(mapping, n, invol_type)

    return mapping, signs


def map_irrep_to_matrices(mapping, signs, n, invol_type):
    return {op: signs[node] * E(node, n, invol_type) for node, op in mapping.items()}


"""
def irrep_dot(coeffs, generators, mapping, signs, n, invol_type):            
    out = 0.
    inv_mapping = {op: node for node, op in mapping.items() if op in generators}
    for c, gen in zip(coeffs, generators):
        node = inv_mapping[gen]
        out += c * signs[node] * E(node, n, invol_type)
    return out
"""


def irrep_dot(coeffs, generators, mapping, n, invol_type):
    out = 0.0
    inv_mapping = {op: (node, sign) for node, (op, sign) in mapping.items() if op in generators}
    for c, gen in zip(coeffs, generators):
        node, sign = inv_mapping[gen]
        out += c * sign * E(node, n, invol_type)
    return out


def map_matrix_to_reducible(matrix, mapping, signs, invol_type):
    assert invol_type == "BDI"
    op = {}
    for i, j in zip(*np.where(matrix)):
        if i < j:
            op[mapping[(i, j)]] = matrix[i, j] / 2 / signs[(i, j)]

    return PauliSentence(op)


def E(node, n, invol_type):
    if invol_type == "BDI":
        e = np.zeros((n, n))
        i, j = node
        e[i, j] = 2
        e[j, i] = -2
    if invol_type == "DIII":
        e = np.zeros((n, n))
        i, j, t, s = node
        sign = 1 if s == "+" else -1
        if t == "A":
            e[i, j] = 1
            e[j, i] = -1
            e[i + n // 2, j + n // 2] = sign
            e[j + n // 2, i + n // 2] = -sign
        if t == "B":
            e[i, j + n // 2] = 1
            e[j + n // 2, i] = -1
            e[j, i + n // 2] = sign
            e[i + n // 2, j] = -sign

    if invol_type == "AIII":
        e = np.zeros((n, n))
        i, j, t = node
        if t == "X":
            e[i, j] = e[j, i] = 1j
        if t == "Y":
            e[i, j] = 1
            e[j, i] = -1
        if t == "Z":
            e[i, i] = 1
            e[i + 1, i + 1] = -1

    return e


def make_so_2n(n):
    """Create all Pauli words for the reducible so(2n) representation implemented
    by the transverse field XY model, i.e., generated by XX couplings, YY couplings, and
    single-qubit Z operators."""
    algebra = [
        PauliWord({w: P1, v: P2} | {i: "Z" for i in range(w + 1, v)})
        for w, v in combinations(range(n), r=2)
        for P1, P2 in product("XY", repeat=2)
    ]
    algebra += [PauliWord({w: "Z"}) for w in range(n)]
    return algebra


def make_so_2n_horizontal_mapping(n):
    """Create a default reducible-to-irreducible mapping for some horizontal operators
    in a BDI decomposition of so(2n). The mapped operators are XX couplings, YY couplings,
    and single-qubit Z operators."""
    mapping = {(i, i + n): PauliWord({i: "Z"}) for i in range(n)}
    mapping |= {(i, i + n + 1): PauliWord({i: "X", i + 1: "X"}) for i in range(n - 1)}
    mapping |= {(i, i + n - 1): PauliWord({i - 1: "Y", i: "Y"}) for i in range(1, n)}
    return mapping


def make_so_2n_full_mapping(n, xy_symmetric=False):
    """Create a default reducible-to-irreducible mapping for all operators of so(2n) implemented
    by the transverse field XY model. The irrep is compatible with a BDI decomposition and
    the Hamiltonian terms (XX coupling, YY coupling, Z field) to be horizontal. In particular
    the mapping created by `make_so_2n_horizontal_mapping` is contained in the mapping created
    here."""
    if xy_symmetric:
        return _so_2n_full_mapping_xy(n)

    mapping = {}
    # upper left triangle
    mapping |= {
        (i, j): PauliWord({i: "X", j: "Y"} | {w: "Z" for w in range(i + 1, j)})
        for i in range(n - 1)
        for j in range(i + 1, n)
    }
    # lower right triangle
    mapping |= {
        (n + i, n + j): PauliWord({i: "Y", j: "X"} | {w: "Z" for w in range(i + 1, j)})
        for i in range(n - 1)
        for j in range(i + 1, n)
    }
    # upper right triangle of off-diagonal
    mapping |= {
        (i, n + j): PauliWord({i: "X", j: "X"} | {w: "Z" for w in range(i + 1, j)})
        for i in range(n - 1)
        for j in range(i + 1, n)
    }
    # lower left triangle of off-diagonal
    mapping |= {
        (j, n + i): PauliWord({i: "Y", j: "Y"} | {w: "Z" for w in range(i + 1, j)})
        for i in range(n - 1)
        for j in range(i + 1, n)
    }
    # diagonal of off-diagonal
    mapping |= {(i, n + i): PauliWord({i: "Z"}) for i in range(n)}

    signs = {(i, n + i): -1 for i in range(n)}
    signs |= {(i, j): 1 for i in range(n) for j in range(i + 1, 2 * n)}
    signs |= {(i, j): -1 for i in range(n, 2 * n) for j in range(i, 2 * n)}
    return mapping, signs


def make_so_2n_full_mapping_str(n, xy_symmetric=False):
    """Create a default reducible-to-irreducible mapping for all operators of so(2n) implemented
    by the transverse field XY model. The irrep is compatible with a BDI decomposition and
    the Hamiltonian terms (XX coupling, YY coupling, Z field) to be horizontal. In particular
    the mapping created by `make_so_2n_horizontal_mapping` is contained in the mapping created
    here."""
    if xy_symmetric:
        raise ValueError
        return _so_2n_full_mapping_xy_str(n)

    mapping = {}
    # upper left triangle
    mapping |= {
        (i, j): (f"{i}X{j - i - 1}Y{n-j-1}", 1) for i in range(n - 1) for j in range(i + 1, n)
    }
    # lower right triangle
    mapping |= {
        (n + i, n + j): (f"{i}Y{j - i - 1}X{n-j-1}", -1)
        for i in range(n - 1)
        for j in range(i + 1, n)
    }
    # upper right triangle of off-diagonal
    mapping |= {
        (i, n + j): (f"{i}X{j - i - 1}X{n-j-1}", 1) for i in range(n - 1) for j in range(i + 1, n)
    }
    # lower left triangle of off-diagonal
    mapping |= {
        (j, n + i): (f"{i}Y{j - i - 1}Y{n-j-1}", 1) for i in range(n - 1) for j in range(i + 1, n)
    }
    # diagonal of off-diagonal
    mapping |= {(i, n + i): (f"{i}Z{n - i - 1}", -1) for i in range(n)}

    # signs = {(i, n+i): -1 for i in range(n)}
    # signs |= {(i, j): 1 for i in range(n) for j in range(i+1, 2 * n)}
    # signs |= {(i, j): -1 for i in range(n, 2*n) for j in range(i, 2*n)}
    return mapping


def _make_tfXY_coeffs(n, coefficients):
    if coefficients == "random":
        alphas = np.random.normal(0.6, 1.0, size=n - 1)
        betas = np.random.normal(0.3, 1.2, size=n - 1)
        gammas = np.random.normal(0.0, 0.3, size=n)
    elif coefficients == "random TF":
        alphas = np.ones(n - 1)
        betas = np.ones(n - 1)
        gammas = np.random.normal(0.0, 0.3, size=n)
    elif coefficients == "uniform":
        alphas = np.ones(n - 1)
        betas = np.ones(n - 1)
        gammas = np.ones(n)

    norm = np.linalg.norm(np.concatenate([alphas, betas, gammas]))
    return alphas / norm, betas / norm, gammas / norm


def make_tfXY_hamiltonian_irrep(n, coefficients="random"):
    """Create the transverse-field XY model Hamiltonian on n qubits,
    represented in a free-fermionic picture as 2n x 2n matrix.
    This function uses the hardcoded mapping from the manuscript for this particular
    model.
    The Hamiltonian is normalized to trace norm 1.
    """
    alphas, betas, gammas = _make_tfXY_coeffs(n, coefficients)
    H_irrep = np.diag(-gammas, k=n) - np.diag(-gammas, k=-n)
    H_irrep += np.diag(alphas, k=n + 1) - np.diag(alphas, k=-n - 1)
    _betas = np.concatenate([[0], betas, [0]])
    H_irrep += np.diag(_betas, k=n - 1) - np.diag(_betas, k=-n + 1)
    return H_irrep


def make_tfXY_hamiltonian_qubits(n, coefficients="random"):
    """Create the transverse-field XY model Hamiltonian on n qubits,
    in its original representation on qubits.
    The Hamiltonian is normalized to trace norm 1.
    """
    alphas, betas, gammas = _make_tfXY_coeffs(n, coefficients)
    coeffs = np.concatenate([alphas, betas, gammas])
    couplings = [X(w) @ X(w + 1) for w in range(n - 1)] + [Y(w) @ Y(w + 1) for w in range(n - 1)]
    Zs = [Z(w) for w in range(n)]
    generators = couplings + Zs
    H = qml.dot(coeffs, generators)
    generators = [next(iter(op.pauli_rep)) for op in generators]
    return H, generators, coeffs
