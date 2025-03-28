import pytest
import numpy as np
from itertools import combinations
import pennylane as qml
from pennylane.pauli import PauliWord
from pennylane.labs.dla import structure_constants_dense
from kak_tools import lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices


gens_for_closure = [
    [PauliWord({0: "X"}), PauliWord({1: "X"}), PauliWord({0: "Z", 1: "Z"})],
    [PauliWord({0: "X"}), PauliWord({0: "Y"})],
    [PauliWord({0: "X"}), PauliWord({1: "Y"})],
    (
        [PauliWord({i: "X", i + 1: "Y"}) for i in range(2)]
        + [PauliWord({i: "Y", i + 1: "Z"}) for i in range(2)]
        + [PauliWord({i: "Z"}) for i in range(3)]
    ),
    (
        [PauliWord({i: "X", i + 1: "X"}) for i in range(5)]
        + [PauliWord({i: "Y", i + 1: "Y"}) for i in range(5)]
        + [PauliWord({i: "Z"}) for i in range(6)]
    ),
]


class TestHelperMethods:

    @pytest.mark.parametrize("generators", gens_for_closure)
    def test_lie_closure_pauli_words(self, generators):
        """Test that Lie closure using Pauli words matches PL's method."""

        pw_closure = lie_closure_pauli_words(generators)
        closure = [next(iter(op.pauli_rep)) for op in qml.lie_closure(generators)]

        assert len(set(pw_closure)) == len(pw_closure)
        assert len(set(closure)) == len(closure)
        assert set(pw_closure) == set(closure)

    @pytest.mark.parametrize("n", [2, 4, 7, 10, 15])
    def test_lie_closure_pauli_words_tfXY(self, n):
        """Test that the DLA of the transverse-field XY model is produced correctly."""

        couplings = [qml.X(w) @ qml.X(w + 1) for w in range(n - 1)] + [
            qml.Y(w) @ qml.Y(w + 1) for w in range(n - 1)
        ]
        Zs = [qml.Z(w) for w in range(n)]
        # Map from general qml.operation.Operator instances to qml.pauli.PauliWord instances
        generators = [next(iter(op.pauli_rep)) for op in (couplings + Zs)]

        closure = lie_closure_pauli_words(generators)

        n_so = 2 * n  # The "n" in so(n)
        so_dim = (n_so**2 - n_so) // 2
        assert len(closure) == so_dim


class TestMapSimpleToIrrep:

    @pytest.mark.parametrize("n", [2, 4, 7, 10])
    def test_tfXY_model(self, n):
        """Test map_simpl_to_irrep with transverse-field XY model."""
        couplings = [qml.X(w) @ qml.X(w + 1) for w in range(n - 1)] + [
            qml.Y(w) @ qml.Y(w + 1) for w in range(n - 1)
        ]
        Zs = [qml.Z(w) for w in range(n)]
        # Map from general qml.operation.Operator instances to qml.pauli.PauliWord instances
        generators = [next(iter(op.pauli_rep)) for op in (couplings + Zs)]
        ops = lie_closure_pauli_words(generators)
        mapping, signs = map_simple_to_irrep(ops, generators, n=2 * n, invol_type="BDI")
        rev_mapping = {val: key for key, val in mapping.items()}
        for op1, op2 in combinations(ops, r=2):
            op_com = (1j * op1).commutator(1j * op2) / 1j
            op_com.simplify()
            ids1 = rev_mapping[op1]
            ids2 = rev_mapping[op2]
            if ids1[0] == ids2[0]:
                com_sgn = signs[ids1] * signs[ids2] * (-1)
                if ids1[1] > ids2[1]:
                    com_ids = (ids2[1], ids1[1])
                    com_sgn *= -1
                else:
                    com_ids = (ids1[1], ids2[1])
            elif ids1[0] == ids2[1]:
                com_ids = (ids2[0], ids1[1])
                com_sgn = -1 * signs[ids1] * signs[ids2]
            elif ids1[1] == ids2[0]:
                com_ids = (ids1[0], ids2[1])
                com_sgn = signs[ids1] * signs[ids2]
            elif ids1[1] == ids2[1]:
                com_sgn = signs[ids1] * signs[ids2] * (-1)
                if ids1[0] > ids2[0]:
                    com_ids = (ids2[0], ids1[0])
                    com_sgn *= -1
                else:
                    com_ids = (ids1[0], ids2[0])
            else:
                assert len(op_com) == 0
                continue
            com_map = 2 * com_sgn * mapping[com_ids] / signs[com_ids]
            assert qml.equal(com_map, op_com), f"{op1, op2}"


class TestMapIrrepToMatrices:

    @pytest.mark.parametrize("n", [2, 4, 7, 8])
    def test_tfXY_model(self, n):
        """Test map_irrep_to_matrices with transverse-field XY model."""
        couplings = [qml.X(w) @ qml.X(w + 1) for w in range(n - 1)] + [
            qml.Y(w) @ qml.Y(w + 1) for w in range(n - 1)
        ]
        Zs = [qml.Z(w) for w in range(n)]
        # Map from general qml.operation.Operator instances to qml.pauli.PauliWord instances
        generators = [next(iter(op.pauli_rep)) for op in (couplings + Zs)]
        ops = lie_closure_pauli_words(generators)
        mapping, signs = map_simple_to_irrep(ops, generators, n=2 * n, invol_type="BDI")
        matrix_map = map_irrep_to_matrices(mapping, signs, 2 * n, invol_type="BDI")

        rev_mapping = {val: key for key, val in mapping.items()}
        for op1, op2 in combinations(ops, r=2):
            op_com = (1j * op1).commutator(1j * op2) / 1j
            op_com.simplify()
            mat1 = matrix_map[op1]
            mat2 = matrix_map[op2]
            mat_com = mat1 @ mat2 - mat2 @ mat1
            if len(op_com) == 0:
                assert np.allclose(mat_com, 0.0)
                continue
            [(com_pw, com_coeff)] = op_com.items()
            commutator_from_ops = matrix_map[com_pw] * com_coeff
            assert np.allclose(commutator_from_ops, mat_com)

        all_mats = np.array([matrix_map[op] for op in ops])
        # Convention: Input operators with a missing factors of 1j (Hermitian instead of skew-Hermitian)
        adj_irrep = structure_constants_dense(all_mats / 1j, is_orthonormal=False)
        adj_redrep = qml.structure_constants(ops)
        assert np.allclose(adj_irrep, adj_redrep)
