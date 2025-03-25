import jax
jax.config.update("jax_enable_x64", True)

import pytest
import numpy as np
import pennylane as qml
from scipy.linalg import expm
from kak_tools import lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices, recursive_bdi, map_recursive_decomp_to_reducible


class TestFDHSCompilation:

    # to do: debug odd-n case
    @pytest.mark.parametrize("n", [2, 4, 7, 8])
    def test_tfXY_model(self, n):
        couplings = [qml.X(w) @ qml.X(w+1) for w in range(n-1)] + [qml.Y(w) @ qml.Y(w+1) for w in range(n-1)]
        Zs = [qml.Z(w) for w in range(n)]
        # Map from general qml.operation.Operator instances to qml.pauli.PauliWord instances
        generators = [next(iter(op.pauli_rep)) for op in (couplings + Zs)]
        ops = lie_closure_pauli_words(generators)
        mapping, signs = map_simple_to_irrep(ops, generators, n=2*n, invol_type="BDI")
        matrix_map = map_irrep_to_matrices(mapping, signs, 2*n, invol_type="BDI")

        alphas = np.ones(n-1)
        betas = np.ones(n-1)
        gammas = np.random.normal(0., 0.3, size=n)
        coeffs = np.concatenate([alphas, betas, gammas])
        coeffs /= np.linalg.norm(coeffs) # Normalization
        H = qml.dot(coeffs, generators)
        terms_irrep = np.stack([matrix_map[gen] for gen in generators])
        H_irrep = np.tensordot(coeffs, terms_irrep, axes=[[0], [0]])

        epsilon = 0.01
        U = expm(epsilon * H_irrep)

        recursive_decomp = recursive_bdi(U, 2 * n, validate=False)
        pauli_decomp = map_recursive_decomp_to_reducible(recursive_decomp, mapping, signs, time=epsilon, validate=False)
        paulirot_decomp = [(coeff, qml.pauli.pauli_word_to_string(pw), pw.wires, _type) for pw, coeff, _type in pauli_decomp]

        def kak_time_evolution(time):
            # invert order because circuits and matrix products are written in opposite order
            for coeff, pauli_str, wires, _type in paulirot_decomp[::-1]:
                # Rescale the coefficient by the evolution time if the Pauli term is in the
                # central Cartan subalgebra
                if _type == "a0":
                    coeff = coeff * time
                # Multiply by (-2) to undo the conventional builtin prefactor of -1/2 for PauliRot
                qml.PauliRot(-2 * coeff, pauli_word=pauli_str, wires=wires)

        N = qml.dot([(1 - r) / 2 for r in range(1, n+1)], Zs) + sum([(r-1)/2 for r in range(1, n+1)])
        N_sq = qml.simplify(N @ N)

        dev_def = qml.device("default.qubit", wires=n) # Slower but supports large dense matrices

        @qml.qnode(dev_def, grad_on_execution=False)
        def kak_circuit(time):
            qml.X(0)
            kak_time_evolution(-time)
            return qml.expval(N_sq)

        times = np.linspace(0, 25, 50)
        kak_out = jax.vmap(kak_circuit, in_axes=0)(times)
        assert np.allclose(np.abs(kak_out), kak_out)
        kak_out = np.abs(kak_out)

        @qml.qnode(dev_def, interface="jax", grad_on_execution=False)
        def exact_circuit(time):
            qml.X(0)
            qml.exp(-1j * time * H)
            return qml.expval(N_sq)

        exact_circuit = jax.jit(exact_circuit)
        exact_out = np.array([exact_circuit(t) for t in times])

        assert np.allclose(exact_out, kak_out)

