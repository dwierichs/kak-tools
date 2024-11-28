"""This file contains tools for handling DLAs made of Pauli words."""

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import rustworkx as rx
import pennylane as qml


def is_int(x):
    return np.isclose(x % 1, 0)


def split_pauli_algebra(dla, verbose=False):
    """Split a list of Pauli words that make up a DLA into multiple sublists
    that make up the connected components of the anticommutation graph of the DLA.
    Note that these components may not be simple by themselves, but can be a semisimple
    algebra with 2^{n_C} isomorphic simple components, c.f., Theorm 2 in
    `Aguilar et al. <https://arxiv.org/pdf/2408.00081>`__.

    Args:
        dla (List[qml.pauli.PauliWord]): List of Pauli words that make up the DLA.
        verbose (bool): Whether or not to print a status report.

    Returns:
        list[set[qml.pauli.PauliWord]]: A list of sets, with each set containing the Pauli
        words that make up a connected component of the anticommutation graph of ``dla``.

    """
    assert all(isinstance(op, qml.pauli.PauliWord) for op in dla)
    # Create fully disconnected graph with Pauli words as nodes
    graph = rx.PyGraph()
    graph.add_nodes_from(dla)

    # Add all anticommutation relations as edges to the graph
    for (i, op1), (j, op2) in combinations(enumerate(dla), r=2):
        if not op1.commutes_with(op2):
            graph.add_edge(i, j, None)

    # Get connected components of the graph. The components are given as collections of indices
    comps = rx.connected_components(graph)
    # Map the groups of indices to groups of Pauli words
    dims = [len(comp) for comp in comps]
    comps_ops = [{dla[i] for i in comp} for comp in comps]
    num_comps = len(comps_ops)
    if num_comps == 1:
        comps_ops = comps_ops[0]
        dims = dims[0]
    if verbose:
        plural = "s" * (num_comps > 1)
        print(f"Found {num_comps} component{plural} with dimension{plural} {dims}.")

    return comps_ops


def get_simple_dim(dla_type, dim):
    """Compute the candidate n for which the algebra of type ``dla_type`` has dimension ``dim``.

    Args:
        dim (int): Dimension of the candidate simple algebra.

    Returns:
        bool: Whether the input dimension matches the dimension of the simple algebra for some n.
        int: The candidate n. The first return value simply tells us whether this is
            close to an integer.
    """
    assert dla_type in ["su", "so", "sp"]

    if dla_type == "su":
        n = np.sqrt(dim + 1)
    elif dla_type == "so":
        n = (np.sqrt(8 * dim + 1) + 1) / 2
    elif dla_type == "sp":
        n = (np.sqrt(8 * dim + 1) - 1) / 4

    if b := is_int(n):
        n = int(np.round(n))
    return b, n


_exceptional_dims = {3: ("so", 3), 10: ("so", 5), 15: ("so", 6)}


def _identify_component(dim):
    """Non-uniquely identify a connected component of the anticommutation graph of a Lie
    algebra made up of Pauli words. The possible identifications are found by the dimension of
    the component.
    """
    # if dim == 6:
    # raise ValueError("Encountered a copy of so(4), which looks simple in the Pauli basis but is not simple. Please handle this manually.")

    candidates = []

    # Handle possible su(n)
    _dim = dim
    factor = 1
    while _dim > 2:
        if _dim in _exceptional_dims:
            candidates.append((factor,) + _exceptional_dims[_dim])

        else:
            dla_type = "su"
            is_of_dla_type, n = get_simple_dim(dla_type, _dim)
            if is_of_dla_type:
                # This could be a su(n) but we know that n must be 2^p with p>=3
                if is_int(np.log2(n)) and np.log2(n) >= 3:
                    candidates.append((factor, dla_type, n))

            is_so, n_so = get_simple_dim("so", _dim)
            is_sp, n_sp = get_simple_dim("sp", _dim)
            if is_so and np.isclose(n_so % 2, 0):
                assert not is_sp
                # If the answer is so(4), we instead count this as 2 so(3) factors (because so(4)
                # is not simple), which is redundant with a different iteration of the while loop.
                if n_so != 4:
                    candidates.append((factor, "so", n_so))
            elif is_so:
                assert is_sp and n_sp == (n_so - 1) // 2
                candidates.extend([(factor, "so", n_so), (factor, "sp", n_sp)])
            else:
                assert not is_sp

        if _dim % 2 == 1:
            break
        _dim = _dim // 2
        factor *= 2

    return candidates


def identify_algebra(comp, verbose=False):
    """Non-uniquely identify a Lie algebra made up of Pauli words by the dimension of the
    connected components of its anticommutation graph.
    """
    if single_comp := all(isinstance(el, qml.pauli.PauliWord) for el in comp):
        components = [comp]
    elif all(
        isinstance(el, Iterable) and all(isinstance(sub_el, qml.pauli.PauliWord) for sub_el in el)
        for el in comp
    ):
        components = comp
    else:
        raise ValueError(
            f"Expected a list of iterables of PauliWords, or a single Iterable of PauliWords, but got\n{comp}"
        )

    results = []
    for i, component in enumerate(components):
        dim = len(component)
        if verbose:
            print(f"Dimension of component: {dim}.")
        candidates = _identify_component(dim)
        if len(candidates) == 0:
            raise ValueError(
                f"Encountered a simple Lie algebra of dimension {dim}, which could not be identified."
            )
        if verbose:
            print(f"Component {i} can be one of the following:")
            for factor, dla_type, n in candidates:
                print(f"{factor} copies of " * (factor > 1) + f"{dla_type}({n})")
        results.append(candidates)

    if single_comp:
        results = results[0]

    return results
