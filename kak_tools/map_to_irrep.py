from itertools import combinations, product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pennylane.pauli import PauliWord
from .pauli_dlas import anticom_graph_pauli

def _anticom_graph_bdi(n, invol_kwargs):
    assert set(invol_kwargs).issubset({"p", "q"})
    if n % 2 or invol_kwargs.get("p", None) != invol_kwargs.get("q", None):
        raise NotImplementedError("BDI currently only is supported with p=q")
    nodes = list(combinations(range(n), r=2))
    nodes_hor = list(product(range(n//2), range(n//2, n)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(n1, n2) for n1, n2 in combinations(graph.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph.add_edges_from(edges)

    graph_hor = nx.Graph()
    graph_hor.add_nodes_from(nodes_hor)
    edges_hor = [(n1, n2) for n1, n2 in combinations(graph_hor.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph_hor.add_edges_from(edges_hor)
    return graph, graph_hor

def _anticom_graph_diii(n, invol_kwargs):
    assert n % 2 == 0
    assert set(invol_kwargs) == set()
    m = n // 2
    # See encoding description somewhere else
    nodes = [(i, j, _type, sign) for _type in "AB" for sign in "+-" for i, j in combinations(range(m), r=2)]
    nodes_hor = [(i, j, _type, "-") for _type in "AB" for i, j in combinations(range(m), r=2)]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(n1, n2) for n1, n2 in combinations(graph.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph.add_edges_from(edges)

    graph_hor = nx.Graph()
    graph_hor.add_nodes_from(nodes_hor)
    edges_hor = [(n1, n2) for n1, n2 in combinations(graph_hor.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph_hor.add_edges_from(edges_hor)
    return graph, graph_hor

def _anticom_graph_aiii(n, invol_kwargs):
    assert set(invol_kwargs).issubset({"p", "q"})
    if n % 2 or invol_kwargs.get("p", None) != invol_kwargs.get("q", None):
        raise NotImplementedError("BDI currently only is supported with p=q")
    m = n // 2
    nodes = [(i, j, t) for t in "XYZ" for i, j in combinations(range(m), r=2) if t!=Z or i==j]
    nodes_hor = [(i, j, t) for t in "XY" for i, j in product(range(m), range(m, n))]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = [(n1, n2) for n1, n2 in combinations(graph.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
    graph.add_edges_from(edges)

    graph_hor = nx.Graph()
    graph_hor.add_nodes_from(nodes_hor)
    edges_hor = [(n1, n2) for n1, n2 in combinations(graph_hor.nodes(), r=2) if n1[0] in n2 or n1[1] in n2]
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
    #hor_subgraph = horizontal_graph.subgraph(list(mapping)).copy()
    return mapping#, hor_subgraph

def _node_commutator(node1, node2, invol_type):
    if invol_type == "BDI":
        a, b = (set(node1) | set(node2)).difference(set(node1) & set(node2))
        if a > b:
            a, b = b, a
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

    """
    elif invol_type == "AIII":
        i, j, t1 = node1
        k, l, t2 = node2
        a, b = {i, j, k, l}.difference({i, j} & {k, l})
        if a > b:
            a, b = b, a
        if t1 == t2:
            assert t1 != "Z"
            return (a, b, "Y")
        if (t1 == "X" and t2 == "Y") or (t1 == "Y" and t2 == "X"):
            if a != b:
                return (a, b, "X")
            return (a, b, 
    """
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



    #edges = list(hor_subgraph.edges())
    #for n1, n2 in edges:
        #com_node = _node_commutator(n1, n2, invol_type)
        #mapping[com_node] = mapping[n1]._commutator(mapping[n2])[0]
        #hor_subgraph.add_node(com_node)
        #single_neighbours = set(hor_subgraph[n1]).symmetric_difference(set(hor_subgraph[n2]))
        #hor_subgraph.add_edges_from([(com_node, n) for n in single_neighbours])
    #return mapping, hor_subgraph

def _yield_sorted_ids_no_collision(i, j, n):
    # Iterate from 0 to n-1 and return sorted versions of (i, k), (k, j)
    for k in range(i):
        yield (k, i), (k, j)
    for k in range(i+1, j):
        yield (i, k), (k, j)
    for k in range(j+1, n):
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


    """
    for _ in range(n_epochs):
        op_added = False
        edges = set(hor_subgraph.edges())
        for n1, n2 in edges.difference(checked_edges):
            com_node = _node_commutator(n1, n2, invol_type)
            if com_node not in hor_subgraph:
                com_pw, _ = mapping[n1]._commutator(mapping[n2])
                mapping[com_node] = com_pw
                missing_ops.remove(com_pw)
                hor_subgraph.add_node(com_node)
                single_neighbours = set(hor_subgraph[n1]).symmetric_difference(set(hor_subgraph[n2]))
                hor_subgraph.add_edges_from([(com_node, n) for n in single_neighbours])
                op_added = True
            checked_edges |= {(n1, n2), (n2, n1), (com_node, n1), (n1, com_node), (com_node, n2), (n2, com_node)} # TODO: Make this a graph
        if not op_added:
            break
    """

    return mapping, missing, missing_ops


def map_choice(mapping, missing, missing_ops, n, invol_type):
    #hor_nodes = set(hor_subgraph.nodes())
    #node = next((0, i) for i in range(1, n) if (0, i) not in hor_nodes)
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
    

def make_signs(mapping, n, invol_type):
    signs = {(0, i): 1 for i in range(1, n)}
    while len(signs) < len(mapping):
        for (n1, pre_sign1), (n2, pre_sign2) in combinations(signs.items(), r=2):
            if not (n1[0] in n2 or n1[1] in n2):
                continue
            node_com_sign = _node_commutator_sign(n1, n2, invol_type)
            com_node = _node_commutator(n1, n2, invol_type)
            pw_com_sign = mapping[n1]._commutator(mapping[n2])[1] / -2j
            relative_sign = node_com_sign * pre_sign1 * pre_sign2 / pw_com_sign
            if signs.get(com_node, relative_sign) != relative_sign:
                raise ValueError("Inconsistency")
            signs[com_node] = relative_sign
    return signs
            

def map_simple_to_irrep(ops, horizontal_ops=None, n=None, invol_type=None, invol_kwargs=None):
    """Map a list of Pauli words that is guaranteed to form a simple Lie algebra of type
    ``dla_type`` to the elements in an irreducible representation of the algebra."""
    assert all(isinstance(op, PauliWord) for op in ops)
    if horizontal_ops is None:
        raise NotImplementedError("This is the simpler scenario, but it is not implemented yet.")
    assert all(isinstance(op, PauliWord) for op in horizontal_ops)

    assert isinstance(n, int)
    assert invol_type in {"AI", "AII", "AIII", "BDI", "CI", "CII", "DIII"}
    if invol_type in {"AII", "DIII"}:
        assert n % 2 == 0
    if invol_kwargs is None:
        invol_kwargs = {}

    full_graph, horizontal_graph = anticom_graph_irrep(n, invol_type, invol_kwargs)
    pauli_graph = anticom_graph_pauli(horizontal_ops)

    mapping = map_horizontal_subgraph(pauli_graph, horizontal_graph)
    mapping = map_hor_com_hor(mapping, pauli_graph, invol_type=invol_type)
    missing = [node for node in full_graph.nodes() if node not in mapping]
    missing_ops = list(set(ops).difference(set(mapping.values())))

    prog_state = (mapping, missing, missing_ops)

    if missing:
        assert missing_ops
        # First completion round
        prog_state = map_coms(*prog_state, n, invol_type=invol_type)

        while prog_state[1]:
            assert prog_state[2]
            prog_state = map_choice(*prog_state, n, invol_type)
            prog_state = map_coms(*prog_state, n, invol_type=invol_type)


    mapping, missing, missing_ops = prog_state
    assert not missing_ops
    assert len(mapping) == len(full_graph)

    signs = make_signs(mapping, n, invol_type)

    return mapping, signs


def map_irrep_to_matrices(mapping, signs, n, invol_type):
    return {op: signs[node] * E(node, n, invol_type) for node, op in mapping.items()}


def E(node, n, invol_type):
    if invol_type == "BDI":
        e = np.zeros((n, n))
        i, j = node
        e[i, j] = 1
        e[j, i] = -1
    if invol_type == "DIII":
        e = np.zeros((n, n))
        i, j, t, s = node
        sign = 1 if s=="+" else -1
        if t == "A":
            e[i, j] = 1
            e[j, i] = -1
            e[i + n//2, j + n//2] = sign
            e[j + n//2, i + n//2] = -sign
        if t == "B":
            e[i, j + n//2] = 1
            e[j + n//2, i] = -1
            e[j, i + n//2] = sign
            e[i + n//2, j] = -sign

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
            e[i+1, i+1] = -1

    return e
