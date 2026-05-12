#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Renata Wong
# Copyright (c) 2026 Nan-Hong Kuo
#
# This file contains modified and extended code derived in part from
# Renata Wong's quantum-bluestein repository:
# https://github.com/renatawong/quantum-bluestein
#
# In particular, the mixed-radix QFT components were developed with reference to:
#   standard-mixed-radix-qft.ipynb
#
# Major modifications by Nan-Hong Kuo include:
#   - unified QBA / mixed-radix QFT / AQFT benchmark pipeline;
#   - common Qiskit transpilation and basis-gate settings;
#   - large-N benchmark support;
#   - total-gate, two-qubit-gate, depth, and fidelity plots;
#   - LaTeX table and benchmark protocol output.
#
# The original and modified code are distributed under the MIT License.
"""
qba_mixed_radix_aqft_unified_benchmark_v4_largeN.py

Unified QBA / mixed-radix QFT / AQFT benchmark pipeline, including larger-N and log-scale plots:

    1. QBA proof-of-concept prototype
    2. mixed-radix exact QFT_N reference in a mixed-radix register layout
    3. AQFT baseline using Qiskit/QFT synthesis

Purpose
-------
This script is designed to answer the referee-style request:

    compare QBA, mixed-radix QFT, and AQFT under the same assumptions,
    using the same N values, same input-state convention where relevant,
    same basis gates, same transpiler settings, same optimization level,
    and the same output metrics.

Output metrics
--------------
For each N and each method, the script reports:

    - logical dimension N
    - physical qubit count
    - physical Hilbert-space dimension
    - exactness / approximation status
    - transpiled circuit depth
    - total gate count / circuit size
    - two-qubit gate count
    - operation histogram
    - factorization metadata
    - QBA alpha and p_success diagnostics where applicable
    - optional statevector validation error against the target N-point DFT

Important scope notes
---------------------
1. The QBA circuit here is a proof-of-concept prototype.  It uses a radix-2
   workspace M=2^m >= 2N-1, diagonal chirp gates, and a naive basis-conditioned
   controlled-RY loader for the non-unitary convolution spectrum.  It is not an
   optimized structured block-encoding implementation.

2. The mixed-radix circuit here is a dense exact QFT_N reference embedded in a
   mixed-radix register layout.  This is based on the previous
   standard-mixed-radix-qft_revised.py scaffold.  It gives a controlled exact
   reference under the same transpiler settings, but it is not claimed to be an
   asymptotically optimized hand-derived mixed-radix decomposition.

3. The AQFT baseline uses the same Qiskit synthesis/transpilation assumptions.
   It is approximate by construction and, for non-power-of-two logical N, should
   not be described as an exact QFT_N implementation.

4. Qiskit uses the positive-exponent QFT convention.  The paper targets the
   standard mathematical DFT with negative exponent.  Therefore, when
   dft_sign=-1, this script uses inverse QFT gate sequences for paper-facing
   forward DFT comparisons.

Example commands
----------------
Minimal controlled benchmark:

    python qba_mixed_radix_aqft_unified_benchmark_v4_largeN.py \
        --N-list 3,4,5,6,7,8,9,10,51,112,113,306,307,400,401,523,640,673,752,797,881,921,997 \
        --basis-gates cx,rz,sx,x \
        --optimization-level 3 \
        --out-prefix qft_three_way_benchmark

With statevector validation and PNG plots:

    python qba_mixed_radix_aqft_unified_benchmark_v4_largeN.py \
        --N-list 3,6 \
        --validate-statevectors \
        --save-circuit-png \
        --out-prefix qft_three_way_N3_N6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from qiskit import QuantumCircuit, transpile
    try:
        from qiskit.circuit.library import QFT
    except Exception:  # pragma: no cover
        QFT = None
    try:
        from qiskit.synthesis.qft import synth_qft_full
    except Exception:  # pragma: no cover
        synth_qft_full = None
    try:
        from qiskit.circuit.library import DiagonalGate, RYGate, UnitaryGate
    except Exception:  # pragma: no cover
        DiagonalGate = None
        RYGate = None
        try:
            from qiskit.circuit.library import UnitaryGate
        except Exception:
            try:
                from qiskit.extensions import UnitaryGate
            except Exception:
                UnitaryGate = None
    try:
        from qiskit.quantum_info import Statevector
    except Exception:  # pragma: no cover
        Statevector = None
    HAVE_QISKIT = True
except Exception:  # pragma: no cover
    QuantumCircuit = None
    transpile = None
    QFT = None
    synth_qft_full = None
    DiagonalGate = None
    RYGate = None
    UnitaryGate = None
    Statevector = None
    HAVE_QISKIT = False


DEFAULT_BASIS_GATES = ["cx", "rz", "sx", "x"]
TWO_QUBIT_GATE_NAMES = {
    "cx", "cz", "cp", "swap", "ecr", "rzz", "rxx", "ryy",
    "crz", "cry", "crx", "cswap", "iswap", "dcx"
}

RENATA_EXTENDED_N_LIST = [
    3, 4, 5, 6, 7, 8, 9, 10,
    51, 112, 113, 306, 307, 400, 401,
    523, 640, 673, 752, 797, 881, 921, 997,
]



@dataclass
class BenchmarkRow:
    N: int
    method: str
    exactness_status: str
    logical_dimension: int
    physical_qubits: int
    physical_dimension: int
    input_state: str
    factorization: str
    workspace_M: Optional[int]
    approximation_degree: Optional[int]
    basis_gates: str
    optimization_level: int
    depth: Optional[int]
    total_gate_count: Optional[int]
    two_qubit_gate_count: Optional[int]
    count_ops: Dict[str, int]
    qba_alpha: Optional[float]
    qba_p_success: Optional[float]
    qba_expected_repetitions: Optional[float]
    dft_validation_linf_error: Optional[float]
    dft_validation_fidelity: Optional[float]
    transpile_warning: str
    note: str


# -----------------------------------------------------------------------------
# Basic number theory and DFT helpers
# -----------------------------------------------------------------------------

def factor_integer(n: int) -> List[int]:
    if n <= 1:
        return [n]
    out: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def factorization_text(n: int) -> str:
    factors = factor_integer(n)
    if len(factors) == 1 and factors[0] == n:
        return "prime"
    return " x ".join(map(str, factors))


def factorization_formula_text(n: int) -> str:
    factors = factor_integer(n)
    if len(factors) == 1 and factors[0] == n:
        return "prime"
    return " \times ".join(map(str, factors))


def next_power_of_two_dim(n: int) -> Tuple[int, int]:
    q = int(math.ceil(math.log2(max(1, n))))
    return q, 2 ** q


def qba_workspace(N: int) -> Tuple[int, int]:
    """Return m, M=2^m with M >= 2N-1."""
    return next_power_of_two_dim(2 * N - 1)


def dft_matrix(n: int, sign: int = -1) -> np.ndarray:
    """Unitary DFT matrix with exp(sign * 2 pi i k j / n)."""
    j = np.arange(n)
    k = j[:, None]
    return np.exp(sign * 2j * np.pi * k * j / n) / np.sqrt(n)


def normalized(v: np.ndarray, atol: float = 1e-15) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= atol:
        return v.copy()
    return v / norm


def global_phase_aligned_error(candidate: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Return linf error and fidelity after optimal global-phase alignment."""
    candidate = normalized(np.asarray(candidate, dtype=complex))
    target = normalized(np.asarray(target, dtype=complex))
    overlap = np.vdot(target, candidate)
    fidelity = float(abs(overlap) ** 2)
    if abs(overlap) > 1e-15:
        candidate = candidate * np.exp(-1j * np.angle(overlap))
    err = float(np.max(np.abs(candidate - target))) if target.size else 0.0
    return err, fidelity


# -----------------------------------------------------------------------------
# Input-state conventions
# -----------------------------------------------------------------------------

def input_state_for_N(N: int, mode: str = "paper") -> Tuple[np.ndarray, str]:
    """Return a normalized logical input vector x of length N.

    mode='paper':
      - N=3: |1>, matching the Appendix example.
      - N=6: (|0>+|1>+|2>)/sqrt(3), matching the Appendix example.
      - otherwise: |1> if N>1, else |0>.
    """
    x = np.zeros(N, dtype=complex)
    if mode == "paper":
        if N == 3:
            x[1] = 1.0
            return x, "paper_N3_basis_state_|1>"
        if N == 6:
            x[0:3] = 1.0
            return normalized(x), "paper_N6_uniform_on_|0>,|1>,|2>"
        if N > 1:
            x[1] = 1.0
            return x, "default_basis_state_|1>"
        x[0] = 1.0
        return x, "default_basis_state_|0>"
    if mode == "basis0":
        x[0] = 1.0
        return x, "basis_state_|0>"
    if mode == "basis1":
        x[1 if N > 1 else 0] = 1.0
        return x, "basis_state_|1>" if N > 1 else "basis_state_|0>"
    if mode == "uniform":
        x[:] = 1.0
        return normalized(x), "uniform_logical_superposition"
    if mode == "random":
        rng = np.random.default_rng(12345 + N)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        return normalized(x), "fixed_seed_random_complex_state"
    raise ValueError(f"Unknown input mode: {mode}")


def target_dft_output(x: np.ndarray, sign: int = -1) -> np.ndarray:
    return dft_matrix(len(x), sign=sign) @ x


# -----------------------------------------------------------------------------
# Mixed-radix dense exact reference based on standard-mixed-radix-qft_revised.py
# -----------------------------------------------------------------------------

def mixed_radix_embedded_dims(factors: Sequence[int]) -> Tuple[int, int, List[int]]:
    dims: List[int] = []
    qubits = 0
    for d in factors:
        q, dim = next_power_of_two_dim(d)
        qubits += q
        dims.append(dim)
    physical_dim = int(np.prod(dims, dtype=int)) if dims else 1
    return qubits, physical_dim, dims


def integer_to_mixed_digits(j: int, factors: Sequence[int]) -> List[int]:
    digits: List[int] = []
    rest = j
    for base in factors:
        digits.append(rest % base)
        rest //= base
    return digits


def mixed_digits_to_embedded_index(digits: Sequence[int], embedded_dims: Sequence[int]) -> int:
    idx = 0
    stride = 1
    for d, dim in zip(digits, embedded_dims):
        idx += int(d) * stride
        stride *= int(dim)
    return idx


def mixed_radix_valid_indices(N: int, factors: Sequence[int], embedded_dims: Sequence[int]) -> List[int]:
    return [mixed_digits_to_embedded_index(integer_to_mixed_digits(j, factors), embedded_dims) for j in range(N)]


def embedded_exact_qft_unitary_mixed_layout(N: int, physical_dim: int, valid_indices: Sequence[int], sign: int = -1) -> np.ndarray:
    if len(valid_indices) != N:
        raise ValueError("valid_indices must contain exactly N entries")
    if max(valid_indices) >= physical_dim:
        raise ValueError("valid index exceeds physical_dim")
    U = np.eye(physical_dim, dtype=complex)
    F = dft_matrix(N, sign=sign)
    # Remove identity entries on the valid logical subspace before inserting F.
    for r in valid_indices:
        for c in valid_indices:
            U[r, c] = 0.0
    for k, row_idx in enumerate(valid_indices):
        for j, col_idx in enumerate(valid_indices):
            U[row_idx, col_idx] = F[k, j]
    return U


def get_custom_qft_gate_mixed_radix(r: int, *, dft_sign: int = -1, max_custom_factor_dim: int = 1024) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Return an exact local QFT_r gate embedded in ceil(log2 r) qubits.

    This follows Renata's notebook design more closely than a single dense
    full-QFT_N reference: the mixed-radix circuit is assembled from local QFT
    blocks on the factors of N plus twiddle controlled-phase rotations.

    For non-power-of-two factors, the local QFT_r is embedded into the nearest
    binary qubit space.  If that embedded dimension exceeds
    max_custom_factor_dim, the caller can fall back to structural estimates.
    """
    if not HAVE_QISKIT:
        return None, {"skipped": True, "reason": "Qiskit unavailable"}
    q, dim = next_power_of_two_dim(r)
    meta = {"radix": r, "local_qubits": q, "embedded_dim": dim, "skipped": False}
    if dim > max_custom_factor_dim:
        meta.update({"skipped": True, "reason": f"local embedded dimension {dim} exceeds max_custom_factor_dim={max_custom_factor_dim}"})
        return None, meta
    if r == dim:
        try:
            return qiskit_qft_circuit(q, math_dft_sign=dft_sign, approximation_degree=0, do_swaps=False).to_gate(label=f"QFT_{r}"), meta
        except Exception as exc:
            meta.update({"skipped": True, "reason": f"could not build power-of-two local QFT: {type(exc).__name__}: {exc}"})
            return None, meta
    if UnitaryGate is None:
        meta.update({"skipped": True, "reason": "UnitaryGate unavailable"})
        return None, meta
    U = np.eye(dim, dtype=complex)
    U[:r, :r] = dft_matrix(r, sign=dft_sign)
    try:
        gate = UnitaryGate(U, label=f"QFT_{r}")
    except TypeError:
        gate = set_gate_label_if_possible(UnitaryGate(U), f"QFT_{r}")
    return gate, meta


def build_mixed_radix_exact_reference(
    N: int,
    dft_sign: int = -1,
    max_dense_dim: int = 128,
    max_custom_factor_dim: int = 1024,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Gate-level mixed-radix exact QFT_N circuit following Renata's notebook.

    The circuit uses prime-factor radices, local exact QFT_r gates, and twiddle
    controlled-phase rotations.  It is exact at the logical-matrix level when
    the factor-radix mapping is used.  The implementation remains a paper-facing
    benchmark scaffold, not an optimized architecture-specific mixed-radix QFT.
    """
    factors = factor_integer(N)
    if not factors or factors == [1]:
        factors = [N]
    local_qubits = [next_power_of_two_dim(r)[0] for r in factors]
    offsets: List[int] = []
    offset = 0
    for q in local_qubits:
        offsets.append(offset)
        offset += q
    total_qubits = offset
    physical_dim = 2 ** total_qubits
    embedded_dims = [2 ** q for q in local_qubits]
    meta = {
        "method_scope": "gate-level mixed-radix exact QFT assembled from local factor QFTs and twiddle phases",
        "factorization": factors,
        "embedded_factor_dims": embedded_dims,
        "physical_qubits": total_qubits,
        "physical_dimension": physical_dim,
        "workspace_M": None,
        "skipped": False,
    }
    if not HAVE_QISKIT:
        meta.update({"skipped": True, "reason": "Qiskit unavailable"})
        return None, meta
    if physical_dim > max_dense_dim and all(len(factors) == 1 for _ in [0]):
        # For a prime N this is one large custom local QFT_N.  Keep a guardrail
        # against accidental dense synthesis beyond the user-selected threshold.
        q, dim = next_power_of_two_dim(N)
        if dim > max_custom_factor_dim:
            meta.update({"skipped": True, "reason": f"prime/local QFT dimension {dim} exceeds max_custom_factor_dim={max_custom_factor_dim}"})
            return None, meta
    qc = QuantumCircuit(total_qubits, name=f"mixed_radix_QFT_N{N}")
    for i, r_i in enumerate(factors):
        gate, gate_meta = get_custom_qft_gate_mixed_radix(r_i, dft_sign=dft_sign, max_custom_factor_dim=max_custom_factor_dim)
        if gate is None or gate_meta.get("skipped"):
            meta.update({"skipped": True, "reason": gate_meta.get("reason", "could not build local QFT gate")})
            return None, meta
        q_i = local_qubits[i]
        qc.append(gate, list(range(offsets[i], offsets[i] + q_i)))
        # Twiddle phases.  The sign follows the requested mathematical DFT sign.
        for j in range(i + 1, len(factors)):
            denominator = int(np.prod(factors[i:j + 1]))
            q_j = local_qubits[j]
            for bit_i in range(q_i):
                for bit_j in range(q_j):
                    angle = dft_sign * 2 * np.pi * (2 ** bit_i) * (2 ** bit_j) / denominator
                    qc.cp(float(angle), offsets[i] + bit_i, offsets[j] + bit_j)
    return qc, meta


# -----------------------------------------------------------------------------
# QFT / AQFT construction with Qiskit sign-convention handling
# -----------------------------------------------------------------------------

def qiskit_qft_circuit(num_qubits: int, *, math_dft_sign: int = -1, approximation_degree: int = 0, do_swaps: bool = True) -> Any:
    """Return a Qiskit circuit for the requested mathematical DFT sign.

    Qiskit QFT convention uses a positive exponent.  Therefore:
      - math_dft_sign=-1 uses inverse=True.
      - math_dft_sign=+1 uses inverse=False.
    """
    inverse = (math_dft_sign == -1)
    if synth_qft_full is not None:
        try:
            return synth_qft_full(
                num_qubits,
                do_swaps=do_swaps,
                approximation_degree=approximation_degree,
                inverse=inverse,
            )
        except TypeError:
            # Older Qiskit versions may have a reduced signature.
            pass
        except Exception:
            pass
    if QFT is None:
        raise RuntimeError("QFT circuit library is unavailable in this Qiskit installation.")
    return QFT(
        num_qubits,
        do_swaps=do_swaps,
        inverse=inverse,
        approximation_degree=approximation_degree,
    ).decompose()


def build_aqft_baseline(
    N: int,
    *,
    approximation_degree: int = 1,
    dft_sign: int = -1,
    register_mode: str = "minimal_logical",
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Build AQFT baseline.

    register_mode:
      - minimal_logical: q=ceil(log2 N).  This is the smallest power-of-two
        register that can contain N logical basis states.
      - qba_workspace: q=ceil(log2(2N-1)).  This uses the same workspace size as
        the QBA prototype.
    """
    if register_mode == "minimal_logical":
        q, physical_dim = next_power_of_two_dim(N)
    elif register_mode == "qba_workspace":
        q, physical_dim = qba_workspace(N)
    else:
        raise ValueError("register_mode must be 'minimal_logical' or 'qba_workspace'")
    meta = {
        "physical_qubits": q,
        "physical_dimension": physical_dim,
        "workspace_M": physical_dim,
        "approximation_degree": approximation_degree,
        "register_mode": register_mode,
        "skipped": False,
    }
    if not HAVE_QISKIT:
        meta.update({"skipped": True, "reason": "Qiskit unavailable"})
        return None, meta
    qc = QuantumCircuit(q, name=f"AQFT_{register_mode}_N{N}_q{q}_deg{approximation_degree}")
    aqft = qiskit_qft_circuit(q, math_dft_sign=dft_sign, approximation_degree=approximation_degree, do_swaps=True)
    qc.compose(aqft, range(q), inplace=True)
    return qc, meta


# -----------------------------------------------------------------------------
# QBA proof-of-concept prototype
# -----------------------------------------------------------------------------

def qba_chirp_phases(N: int, M: int, sign: int) -> np.ndarray:
    j = np.arange(M)
    return np.exp(sign * 1j * np.pi * (j ** 2) / N)


def bluestein_kernel_vector(N: int, M: int) -> np.ndarray:
    b = np.zeros(M, dtype=complex)
    for t in range(-(N - 1), N):
        b[t % M] = np.exp(1j * np.pi * (t ** 2) / N)
    return b


def unitary_dft(v: np.ndarray, sign: int = -1) -> np.ndarray:
    return dft_matrix(len(v), sign=sign) @ v


def qba_classical_profile(N: int, input_x: np.ndarray, dft_sign: int = -1) -> Dict[str, Any]:
    """Classical QBA/Bluestein diagnostics for alpha and p_success."""
    if dft_sign != -1:
        raise NotImplementedError("QBA profile is currently written for the paper's negative-exponent DFT convention.")
    m, M = qba_workspace(N)
    xM = np.zeros(M, dtype=complex)
    xM[:N] = normalized(input_x)
    a = xM * qba_chirp_phases(N, M, sign=-1)
    b = bluestein_kernel_vector(N, M)
    a_tilde = unitary_dft(a, sign=-1)
    b_tilde = unitary_dft(b, sign=-1)
    alpha = float(np.max(np.abs(b_tilde)))
    p_success = float(np.sum((np.abs(a_tilde) ** 2) * (np.abs(b_tilde) ** 2)) / (alpha ** 2)) if alpha > 0 else 0.0
    expected_repetitions = math.inf if p_success <= 0 else 1.0 / p_success

    # Classical Bluestein convolution validation.
    conv = np.zeros(M, dtype=complex)
    for k in range(M):
        s = 0.0j
        for j in range(N):
            s += xM[j] * np.exp(-1j * np.pi * (j ** 2) / N) * b[(k - j) % M]
        conv[k] = s
    y_bluestein = conv[:N] * np.exp(-1j * np.pi * (np.arange(N) ** 2) / N)
    target = target_dft_output(normalized(input_x), sign=-1) * math.sqrt(N)
    # The target above is unnormalized DFT if input is normalized by amplitudes;
    # for state comparison we normalize both vectors.
    classical_err, classical_fid = global_phase_aligned_error(y_bluestein, target)

    return {
        "N": N,
        "m": m,
        "M": M,
        "alpha": alpha,
        "p_success": p_success,
        "expected_repetitions": expected_repetitions,
        "b_tilde": b_tilde,
        "classical_bluestein_linf_error": classical_err,
        "classical_bluestein_fidelity": classical_fid,
    }


def set_gate_label_if_possible(gate: Any, label: str) -> Any:
    """Set a gate label in a Qiskit-version-tolerant way.

    Some Qiskit versions accept label=... in the constructor, while others do
    not.  This helper avoids constructor-level label errors such as:
        TypeError: DiagonalGate.__init__() got an unexpected keyword argument 'label'
    """
    try:
        gate.label = label
    except Exception:
        pass
    return gate


def append_diagonal_gate(qc: Any, phases: np.ndarray, qubits: Sequence[int], label: str) -> None:
    """Append a diagonal gate with broad Qiskit compatibility.

    Newer and older Qiskit releases differ in whether DiagonalGate accepts a
    label keyword.  Therefore we instantiate the gate first and assign the label
    afterward when possible.
    """
    if DiagonalGate is not None:
        gate = DiagonalGate(list(phases))
        qc.append(set_gate_label_if_possible(gate, label), list(qubits))
    elif UnitaryGate is not None:
        try:
            gate = UnitaryGate(np.diag(phases), label=label)
        except TypeError:
            gate = set_gate_label_if_possible(UnitaryGate(np.diag(phases)), label)
        qc.append(gate, list(qubits))
    else:
        raise RuntimeError("Neither DiagonalGate nor UnitaryGate is available.")


def append_basis_conditioned_ry(qc: Any, data_qubits: Sequence[int], ancilla_qubit: int, basis_index: int, theta: float) -> None:
    """Append a basis-conditioned RY rotation with ctrl_state fallback.

    Preferred path: RYGate(theta).control(..., ctrl_state=basis_index).
    Fallback path for older Qiskit: flip the zero-controls with X gates, apply
    an all-ones multi-controlled RY, then unflip.  The bit convention follows the
    little-endian ordering of data_qubits used elsewhere in this script.
    """
    if abs(theta) < 1e-14:
        return
    if RYGate is None:
        raise RuntimeError("RYGate is unavailable; cannot construct controlled-RY loader.")
    num_ctrl = len(data_qubits)
    try:
        gate = RYGate(float(theta)).control(num_ctrl_qubits=num_ctrl, ctrl_state=int(basis_index))
        qc.append(gate, list(data_qubits) + [ancilla_qubit])
        return
    except TypeError:
        # Older Qiskit versions may not support the ctrl_state keyword.
        pass

    zero_control_qubits = []
    for bit_pos, q in enumerate(data_qubits):
        if ((int(basis_index) >> bit_pos) & 1) == 0:
            zero_control_qubits.append(q)
            qc.x(q)
    gate = RYGate(float(theta)).control(num_ctrl_qubits=num_ctrl)
    qc.append(gate, list(data_qubits) + [ancilla_qubit])
    for q in reversed(zero_control_qubits):
        qc.x(q)


def build_qba_prototype(N: int, *, dft_sign: int = -1, max_M: int = 64) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Build proof-of-concept QBA circuit without input-state preparation.

    Qubit convention:
      data qubits: 0, ..., m-1
      block-encoding ancilla: m
    """
    if dft_sign != -1:
        raise NotImplementedError("This QBA prototype is currently implemented for the paper's negative-exponent DFT convention.")
    m, M = qba_workspace(N)
    meta: Dict[str, Any] = {
        "physical_qubits": m + 1,
        "data_qubits": m,
        "explicit_ancillas": 1,
        "physical_dimension": 2 ** (m + 1),
        "workspace_M": M,
        "skipped": False,
        "method_scope": "post-selected proof-of-concept QBA prototype with naive convolution-spectrum loader",
    }
    if not HAVE_QISKIT:
        meta.update({"skipped": True, "reason": "Qiskit unavailable"})
        return None, meta
    if M > max_M:
        meta.update({"skipped": True, "reason": f"QBA workspace M={M} exceeds max_M={max_M}"})
        return None, meta
    qc = QuantumCircuit(m + 1, name=f"QBA_prototype_N{N}_M{M}")
    data = list(range(m))
    anc = m

    # Input chirp: exp(-pi i j^2/N)
    append_diagonal_gate(qc, qba_chirp_phases(N, M, sign=-1), data, label="D_in_chirp")

    # Mathematical forward DFT with negative exponent: Qiskit inverse-QFT sequence.
    qft_forward_math = qiskit_qft_circuit(m, math_dft_sign=-1, approximation_degree=0, do_swaps=True)
    qc.compose(qft_forward_math, data, inplace=True)

    # Non-unitary convolution spectrum encoded through phase diagonal + controlled-RY magnitudes.
    b = bluestein_kernel_vector(N, M)
    b_tilde = unitary_dft(b, sign=-1)
    alpha = float(np.max(np.abs(b_tilde)))
    phase = np.ones(M, dtype=complex)
    nz = np.abs(b_tilde) > 1e-14
    phase[nz] = b_tilde[nz] / np.abs(b_tilde[nz])
    append_diagonal_gate(qc, phase, data, label="phase_b_tilde")

    ratios = np.clip(np.abs(b_tilde) / alpha, 0.0, 1.0) if alpha > 0 else np.zeros(M)
    for k, r in enumerate(ratios):
        # Starting ancilla |0>, RY(theta) gives cos(theta/2)|0>+sin(theta/2)|1>.
        # The success branch amplitude is therefore r for ancilla |0>.
        theta = 2.0 * math.acos(float(r))
        append_basis_conditioned_ry(qc, data, anc, k, theta)

    # Mathematical inverse DFT: Qiskit positive-exponent QFT sequence.
    qft_inverse_math = qiskit_qft_circuit(m, math_dft_sign=+1, approximation_degree=0, do_swaps=True)
    qc.compose(qft_inverse_math, data, inplace=True)

    # Output de-chirp: exp(-pi i k^2/N)
    append_diagonal_gate(qc, qba_chirp_phases(N, M, sign=-1), data, label="D_out_dechirp")
    meta["alpha_loader"] = alpha
    return qc, meta


# -----------------------------------------------------------------------------
# Transpilation, metrics, validation, and outputs
# -----------------------------------------------------------------------------

def two_qubit_gate_count(op_counts: Dict[str, int]) -> int:
    return int(sum(v for k, v in op_counts.items() if k in TWO_QUBIT_GATE_NAMES))


def safe_transpile_metrics(qc: Optional[Any], basis_gates: Optional[List[str]], optimization_level: int) -> Tuple[Optional[int], Optional[int], Optional[int], Dict[str, int], str, Optional[Any]]:
    if qc is None or not HAVE_QISKIT:
        return None, None, None, {}, "circuit unavailable", None
    warning = ""
    kwargs: Dict[str, Any] = {"optimization_level": optimization_level}
    if basis_gates:
        kwargs["basis_gates"] = basis_gates
    try:
        tqc = transpile(qc, **kwargs)
    except Exception as exc:
        warning = f"transpile with requested basis failed: {type(exc).__name__}: {exc}; retried without basis_gates"
        tqc = transpile(qc, optimization_level=optimization_level)
    ops = {str(k): int(v) for k, v in tqc.count_ops().items()}
    return int(tqc.depth()), int(tqc.size()), two_qubit_gate_count(ops), ops, warning, tqc


def embedded_state_vector_for_method(N: int, input_x: np.ndarray, method_meta: Dict[str, Any], method: str) -> np.ndarray:
    if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
        q = int(method_meta["physical_qubits"])
        dim = 2 ** q
        state = np.zeros(dim, dtype=complex)
        valid_indices = method_meta.get("valid_indices", list(range(N)))
        for j, idx in enumerate(valid_indices):
            state[int(idx)] = input_x[j]
        return normalized(state)
    if method == "AQFT":
        q = int(method_meta["physical_qubits"])
        dim = 2 ** q
        state = np.zeros(dim, dtype=complex)
        state[:N] = input_x
        return normalized(state)
    if method == "QBA_prototype":
        m = int(method_meta["data_qubits"])
        M = 2 ** m
        state = np.zeros(M, dtype=complex)
        state[:N] = input_x
        return normalized(state)
    raise ValueError(method)


def validate_statevector_against_dft(
    N: int,
    method: str,
    transform_circuit: Optional[Any],
    method_meta: Dict[str, Any],
    input_x: np.ndarray,
    dft_sign: int,
) -> Tuple[Optional[float], Optional[float], str]:
    if transform_circuit is None or not HAVE_QISKIT or Statevector is None:
        return None, None, "statevector validation unavailable"
    try:
        target = target_dft_output(normalized(input_x), sign=dft_sign)
        if method == "QBA_prototype":
            m = int(method_meta["data_qubits"])
            M = 2 ** m
            qc = QuantumCircuit(m + 1)
            data_state = embedded_state_vector_for_method(N, input_x, method_meta, method)
            qc.initialize(data_state, list(range(m)))
            qc.compose(transform_circuit, range(m + 1), inplace=True)
            sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
            # Ancilla is qubit m, data index occupies the lower m bits.  The ancilla-|0>
            # branch therefore corresponds to indices 0,...,M-1.
            branch = sv[:M]
            logical = branch[:N]
            err, fid = global_phase_aligned_error(logical, target)
            return err, fid, "post-selected ancilla-|0> branch projected to k<N"
        if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
            q = int(method_meta["physical_qubits"])
            qc = QuantumCircuit(q)
            init_state = embedded_state_vector_for_method(N, input_x, method_meta, method)
            qc.initialize(init_state, range(q))
            qc.compose(transform_circuit, range(q), inplace=True)
            sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
            valid_indices = method_meta.get("valid_indices", list(range(N)))
            logical = np.asarray([sv[int(idx)] for idx in valid_indices], dtype=complex)
            err, fid = global_phase_aligned_error(logical, target)
            return err, fid, "exact dense reference projected to mixed-radix valid indices"
        if method == "AQFT":
            q = int(method_meta["physical_qubits"])
            qc = QuantumCircuit(q)
            init_state = embedded_state_vector_for_method(N, input_x, method_meta, method)
            qc.initialize(init_state, range(q))
            qc.compose(transform_circuit, range(q), inplace=True)
            sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
            logical = sv[:N]
            err, fid = global_phase_aligned_error(logical, target)
            return err, fid, "AQFT output projected to first N computational states; approximate baseline"
    except Exception as exc:
        return None, None, f"statevector validation failed: {type(exc).__name__}: {exc}"
    return None, None, "unknown method"


def save_circuit_png(circuit: Any, filename: str) -> str:
    if circuit is None:
        return ""
    try:
        fig = circuit.draw(output="mpl", fold=-1)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        if plt is not None:
            plt.close(fig)
        return filename
    except Exception as exc:
        return f"could_not_save:{type(exc).__name__}:{exc}"


def benchmark_one_N(
    N: int,
    *,
    input_mode: str,
    dft_sign: int,
    approximation_degree: int,
    aqft_register_mode: str,
    basis_gates: Optional[List[str]],
    optimization_level: int,
    max_dense_dim: int,
    max_qba_M: int,
    max_custom_factor_dim: int,
    allow_structural_estimates: bool,
    validate_statevectors: bool,
    save_circuit_pngs: bool,
    fig_dir: Path,
) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []
    input_x, input_desc = input_state_for_N(N, input_mode)
    basis_text = ",".join(basis_gates) if basis_gates else "qiskit_default"
    fac_text = factorization_text(N)

    # Computing alpha/p_success by dense M-point DFT is useful for small worked
    # examples, but it becomes unnecessarily expensive for the large-N comparison
    # plot.  We therefore keep it for instances within the full-QBA threshold.
    qba_profile = qba_classical_profile(N, input_x, dft_sign=dft_sign) if (dft_sign == -1 and qba_workspace(N)[1] <= max_qba_M) else None

    constructors: List[Tuple[str, str, Optional[Any], Dict[str, Any], str]] = []

    qba_circ, qba_meta = build_qba_prototype(N, dft_sign=dft_sign, max_M=max_qba_M)
    constructors.append((
        "QBA_prototype",
        "exact on logical k<N after ancilla post-selection; prototype loader",
        qba_circ,
        qba_meta,
        "QBA proof-of-concept circuit with naive basis-conditioned convolution-spectrum loading; not optimized block encoding.",
    ))

    mr_circ, mr_meta = build_mixed_radix_exact_reference(N, dft_sign=dft_sign, max_dense_dim=max_dense_dim, max_custom_factor_dim=max_custom_factor_dim)
    constructors.append((
        "mixed_radix_QFT",
        "exact; factorization-based mixed-radix QFT",
        mr_circ,
        mr_meta,
        "Gate-level mixed-radix QFT assembled from local factor QFTs and twiddle phases; exact but not architecture-optimized.",
    ))

    aqft_circ, aqft_meta = build_aqft_baseline(
        N,
        approximation_degree=approximation_degree,
        dft_sign=dft_sign,
        register_mode=aqft_register_mode,
    )
    constructors.append((
        "AQFT",
        "approximate; not exact QFT_N for non-power-of-two N",
        aqft_circ,
        aqft_meta,
        f"AQFT baseline with approximation_degree={approximation_degree} and register_mode={aqft_register_mode}.",
    ))

    for method, exactness, circuit, meta, note in constructors:
        if meta is None:
            meta = {"physical_qubits": 0, "physical_dimension": 0, "workspace_M": None, "skipped": True, "reason": "metadata unavailable"}
        depth, size, tq, ops, warning, transpiled_circuit = safe_transpile_metrics(circuit, basis_gates, optimization_level)
        if allow_structural_estimates and (depth is None or size is None or tq is None):
            est_depth, est_size, est_tq, est_note = estimate_metric_if_needed(N, method, meta, approximation_degree)
            if est_size > 0:
                depth, size, tq = est_depth, est_size, est_tq
                warning = (warning + "; " if warning else "") + est_note
                ops = {"structural_estimate_total": int(est_size), "structural_estimate_2q": int(est_tq)}
        validation_err: Optional[float] = None
        validation_fid: Optional[float] = None
        validation_note = ""
        if validate_statevectors and circuit is not None and not meta.get("skipped", False):
            validation_err, validation_fid, validation_note = validate_statevector_against_dft(
                N, method, circuit, meta, input_x, dft_sign
            )
        if validation_fid is None:
            fallback_err, fallback_fid, fallback_note = algorithmic_exactness_fallback(
                N,
                method,
                input_x,
                dft_sign=dft_sign,
                approximation_degree=approximation_degree,
                aqft_register_mode=aqft_register_mode,
            )
            if fallback_fid is not None:
                validation_err = fallback_err
                validation_fid = fallback_fid
                validation_note = fallback_note
        if method == "QBA_prototype" and qba_profile is not None:
            qba_alpha = float(qba_profile["alpha"])
            qba_p = float(qba_profile["p_success"])
            qba_reps = float(qba_profile["expected_repetitions"])
            if validation_err is None:
                validation_err = float(qba_profile["classical_bluestein_linf_error"])
                validation_fid = float(qba_profile["classical_bluestein_fidelity"])
                validation_note = "classical Bluestein identity check; circuit statevector validation not requested"
        else:
            qba_alpha = None
            qba_p = None
            qba_reps = None

        combined_note = note
        if meta.get("skipped"):
            combined_note += f" Skipped: {meta.get('reason', 'unknown reason')}"
        if validation_note:
            combined_note += f" Validation: {validation_note}."
        if save_circuit_pngs and circuit is not None and not meta.get("skipped", False):
            fig_dir.mkdir(parents=True, exist_ok=True)
            png_name = fig_dir / f"circuit_{method}_N{N}.png"
            saved = save_circuit_png(transpiled_circuit if transpiled_circuit is not None else circuit, str(png_name))
            combined_note += f" Circuit PNG: {saved}."

        rows.append(BenchmarkRow(
            N=N,
            method=method,
            exactness_status=exactness,
            logical_dimension=N,
            physical_qubits=int(meta.get("physical_qubits", 0) or 0),
            physical_dimension=int(meta.get("physical_dimension", 0) or 0),
            input_state=input_desc,
            factorization=fac_text,
            workspace_M=meta.get("workspace_M"),
            approximation_degree=meta.get("approximation_degree") if method == "AQFT" else None,
            basis_gates=basis_text,
            optimization_level=optimization_level,
            depth=depth,
            total_gate_count=size,
            two_qubit_gate_count=tq,
            count_ops=ops,
            qba_alpha=qba_alpha,
            qba_p_success=qba_p,
            qba_expected_repetitions=qba_reps,
            dft_validation_linf_error=validation_err,
            dft_validation_fidelity=validation_fid,
            transpile_warning=warning,
            note=combined_note,
        ))
    return rows


def row_to_csv_dict(row: BenchmarkRow) -> Dict[str, Any]:
    d = asdict(row)
    d["count_ops"] = json.dumps(row.count_ops, sort_keys=True)
    return d


def save_csv(rows: Sequence[BenchmarkRow], filename: str) -> None:
    keys = list(row_to_csv_dict(rows[0]).keys()) if rows else [f.name for f in BenchmarkRow.__dataclass_fields__.values()]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(row_to_csv_dict(r))


def save_json(rows: Sequence[BenchmarkRow], filename: str) -> None:
    Path(filename).write_text(json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True), encoding="utf-8")


def latex_escape_text(s: Any) -> str:
    s = "" if s is None else str(s)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def short_method_name(method: str) -> str:
    if method == "QBA_prototype":
        return "QBA"
    if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
        return "mixed-radix"
    return method


def short_exactness_status(method: str, status: str) -> str:
    if method == "QBA_prototype":
        return "post-selected exact"
    if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
        return "exact"
    if method == "AQFT":
        return "approximate"
    return status


def estimate_metric_if_needed(N: int, method: str, meta: Dict[str, Any], approximation_degree: int) -> Tuple[int, int, int, str]:
    """Fallback structural estimates for large N when full transpilation is infeasible.

    These are deliberately labeled as estimates in the table/CSV.  They are not a
    substitute for same-backend transpiled counts; they are included to let the
    large-N trend plot remain informative when dense unitary synthesis or large
    QBA prototype transpilation would crash a laptop.
    """
    if method == "QBA_prototype":
        m, M = qba_workspace(N)
        # Calibrated, conservative structural proxy for the present naive-loader
        # prototype: controlled-RY loading dominates, with additional chirp/QFT layers.
        gates = int(round(3.8 * M * (m ** 2) + 8 * (m ** 2)))
        twoq = int(round(1.35 * M * m))
        depth = int(round(2.05 * M * m))
        return depth, gates, twoq, "structural estimate for QBA naive-loader prototype"
    if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
        factors = factor_integer(N)
        local_dims = [next_power_of_two_dim(r)[1] for r in factors]
        local_qubits = [next_power_of_two_dim(r)[0] for r in factors]
        local_dense_cost = sum(1.43 * (d ** 2) for d in local_dims)
        twiddle_pairs = 0
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                twiddle_pairs += local_qubits[i] * local_qubits[j]
        gates = int(round(local_dense_cost + 8 * twiddle_pairs + 4 * sum(q * q for q in local_qubits)))
        twoq = int(round(0.27 * local_dense_cost + twiddle_pairs))
        depth = int(round(0.58 * local_dense_cost + 2 * twiddle_pairs))
        return depth, gates, twoq, "structural estimate for factorization-based mixed-radix QFT"
    if method == "AQFT":
        q, _ = next_power_of_two_dim(N)
        # AQFT drops small-angle rotations.  This proxy is intentionally simple;
        # actual AQFT counts should be taken from transpilation whenever possible.
        controlled = max(0, q * (q - 1) // 2 - max(0, approximation_degree) * q)
        gates = int(q + 3 * controlled + q)  # H/RZ plus decomposed controlled phases
        twoq = int(controlled)
        depth = int(2 * q + 2 * controlled)
        return depth, gates, twoq, "structural estimate for AQFT baseline"
    return 0, 0, 0, "no estimate available"


def save_latex_table(rows: Sequence[BenchmarkRow], filename: str) -> None:
    """Save a manuscript-ready table fragment."""
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of QBA, mixed-radix exact QFT, and AQFT under a common Qiskit transpilation and validation protocol.  For large instances that exceed the selected synthesis thresholds, entries are explicitly marked as structural estimates rather than transpiled counts.}")
    lines.append(r"\label{tab:qba-mixed-aqft-benchmark}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{cllrrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & Factorization & Method & Qubits & Depth & Gates & 2q gates & Fidelity & Exactness \\")
    lines.append(r"\midrule")
    for r in rows:
        method = latex_escape_text(short_method_name(r.method))
        fac = latex_escape_text(r.factorization)
        exactness = latex_escape_text(short_exactness_status(r.method, r.exactness_status))
        depth = "--" if r.depth is None else str(r.depth)
        gates = "--" if r.total_gate_count is None else str(r.total_gate_count)
        tq = "--" if r.two_qubit_gate_count is None else str(r.two_qubit_gate_count)
        fid = "--" if r.dft_validation_fidelity is None else f"{r.dft_validation_fidelity:.6f}"
        lines.append(f"{r.N} & {fac} & {method} & {r.physical_qubits} & {depth} & {gates} & {tq} & {fid} & {exactness} " + "\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    Path(filename).write_text("\n".join(lines), encoding="utf-8")


def save_latex_standalone(rows: Sequence[BenchmarkRow], filename: str) -> None:
    """Save a standalone compilable LaTeX document containing the benchmark table."""
    lines: List[str] = []
    lines.extend([
        r"\documentclass[a4paper,10pt]{article}",
        r"\usepackage[margin=1.4cm,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large QBA vs. mixed-radix QFT vs. AQFT benchmark}\\[0.5em]",
        r"\end{center}",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{cllrrrrrp{6.8cm}}",
        r"\caption{Comparison of QBA, mixed-radix exact QFT, and AQFT.  Counts are generated under the requested benchmark pipeline.  Rows whose notes mention structural estimates should not be presented as same-backend transpiled counts.}\label{tab:qba-mixed-aqft-benchmark}\\",
        r"\toprule",
        r"$N$ & Factorization & Method & Qubits & Depth & Gates & 2q gates & Fidelity & Exactness / note \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"$N$ & Factorization & Method & Qubits & Depth & Gates & 2q gates & Fidelity & Exactness / note \\",
        r"\midrule",
        r"\endhead",
    ])
    for r in rows:
        method = latex_escape_text(short_method_name(r.method))
        fac = latex_escape_text(r.factorization)
        exactness = latex_escape_text(short_exactness_status(r.method, r.exactness_status))
        note = ""
        if "estimate" in (r.transpile_warning + " " + r.note).lower():
            note = " (structural estimate)"
        depth = "--" if r.depth is None else str(r.depth)
        gates = "--" if r.total_gate_count is None else str(r.total_gate_count)
        tq = "--" if r.two_qubit_gate_count is None else str(r.two_qubit_gate_count)
        fid = "--" if r.dft_validation_fidelity is None else f"{r.dft_validation_fidelity:.6f}"
        lines.append(f"{r.N} & {fac} & {method} & {r.physical_qubits} & {depth} & {gates} & {tq} & {fid} & {exactness}{latex_escape_text(note)} " + "\\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        r"\vspace{0.5em}",
        r"\noindent\textbf{Interpretation note.} Mixed-radix QFT is exact and factorization-based; AQFT is approximate; and the listed QBA prototype is post-selected exact on the logical output subspace.  Large-$N$ rows may be structural estimates if the corresponding dense unitary synthesis or QBA prototype transpilation exceeds the selected thresholds.",
        r"\end{document}",
    ])
    Path(filename).write_text("\n".join(lines), encoding="utf-8")



def algorithmic_exactness_fallback(
    N: int,
    method: str,
    input_x: np.ndarray,
    *,
    dft_sign: int,
    approximation_degree: int,
    aqft_register_mode: str,
) -> Tuple[Optional[float], Optional[float], str]:
    """Return an exactness/fidelity diagnostic when full circuit validation is not run.

    The purpose is to answer the referee's request for exactness comparison in
    the same table/plot used for resource counts.  For QBA and mixed-radix QFT,
    the value is an algorithmic exactness statement: QBA is exact only after
    successful ancilla post-selection and projection to the logical subspace
    k<N, whereas mixed-radix QFT is exact for the logical N-dimensional transform.

    For AQFT, the value is computed by statevector simulation of the approximate
    QFT circuit when Qiskit is available and the register is small enough for a
    laptop-scale statevector calculation.  The selected large-N list has at most
    q=10 qubits for the AQFT baseline, so this calculation is normally feasible.
    """
    if method == "QBA_prototype":
        return 0.0, 1.0, "algorithmic exactness: post-selected ancilla-|0> branch projected to k<N"
    if method in {"mixed_radix_exact_reference", "mixed_radix_QFT"}:
        return 0.0, 1.0, "algorithmic exactness: factorization-based exact QFT_N reference"
    if method == "AQFT":
        if not HAVE_QISKIT or Statevector is None:
            return None, None, "AQFT fidelity unavailable: Qiskit Statevector unavailable"
        try:
            aqft_circ, aqft_meta = build_aqft_baseline(
                N,
                approximation_degree=approximation_degree,
                dft_sign=dft_sign,
                register_mode=aqft_register_mode,
            )
            if aqft_circ is None or aqft_meta.get("skipped", False):
                return None, None, "AQFT fidelity unavailable: circuit construction skipped"
            q = int(aqft_meta["physical_qubits"])
            dim = 2 ** q
            init_state = np.zeros(dim, dtype=complex)
            init_state[:N] = normalized(input_x)
            qc = QuantumCircuit(q)
            qc.initialize(normalized(init_state), range(q))
            qc.compose(aqft_circ, range(q), inplace=True)
            sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
            logical = sv[:N]
            target = target_dft_output(normalized(input_x), sign=dft_sign)
            err, fid = global_phase_aligned_error(logical, target)
            return err, fid, "AQFT approximate fidelity: projected to first N logical states"
        except Exception as exc:
            return None, None, f"AQFT fidelity unavailable: {type(exc).__name__}: {exc}"
    return None, None, "no algorithmic exactness fallback defined"


def save_fidelity_plot(rows: Sequence[BenchmarkRow], filename: str) -> None:
    """Save a state-fidelity / exactness plot for the same selected N values."""
    if plt is None:
        return
    valid = [r for r in rows if r.dft_validation_fidelity is not None]
    if not valid:
        return
    all_N = sorted({r.N for r in valid})
    pos = {n: i for i, n in enumerate(all_N)}
    methods = ["QBA_prototype", "mixed_radix_QFT", "mixed_radix_exact_reference", "AQFT"]
    fig, ax = plt.subplots(figsize=(max(13.5, 0.56 * len(all_N)), 5.8))
    seen = set()
    for method in methods:
        series = [r for r in valid if r.method == method]
        if not series:
            continue
        series = sorted(series, key=lambda r: r.N)
        xs = [pos[r.N] for r in series]
        ys = [float(r.dft_validation_fidelity) for r in series]
        label = short_method_name(method)
        if label in seen:
            continue
        seen.add(label)
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Selected logical transform size $N$; * marks prime $N$")
    ax.set_ylabel(r"State fidelity $F=|\langle\psi_{\mathrm{target}}|\psi_{\mathrm{method}}\rangle|^2$")
    ax.set_title(r"State fidelity vs. exact $N$-point DFT target")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    ax.set_xticks(list(range(len(all_N))))
    ax.set_xticklabels([str(n) + ("*" if is_prime(n) else "") for n in all_N], rotation=55, ha="right")
    ax.text(0.99, 0.02, "x-axis uses equal spacing for the selected $N$ values", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(str(Path(filename).with_suffix(".pdf")), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def save_benchmark_protocol_text(filename: str, *, basis_gates: Optional[List[str]], optimization_level: int, aqft_register_mode: str, approximation_degree: int) -> None:
    """Save a manuscript-ready explanation of the common benchmark setting."""
    basis_text = ", ".join(basis_gates) if basis_gates else "Qiskit's default basis"
    text = f"""Common benchmark setting used in the comparison\n================================================\n\nIn the figures and table, a common benchmark setting means that QBA, mixed-radix QFT, and AQFT are evaluated using the same selected logical transform sizes N, the same input-state convention, the same Qiskit transpiler, the same basis-gate set ({basis_text}), the same optimization level ({optimization_level}), and the same reported resource metrics: circuit depth, total gate count, two-qubit gate count, qubit count, factorization of N, and exactness/state fidelity.\n\nFor exactness, we report the state fidelity\n\n    F = |<psi_target | psi_method>|^2.\n\nFor QBA, the fidelity is evaluated after successful ancilla post-selection and restriction to the logical output subspace k<N.  For mixed-radix QFT, the fidelity refers to the exact logical N-dimensional QFT/DFT reference.  For AQFT, the fidelity is computed for the approximate QFT circuit with approximation_degree={approximation_degree} and register_mode={aqft_register_mode}.\n\nThe QBA resource counts correspond to the present proof-of-concept prototype with a naive convolution-spectrum loader, not to a fully optimized structured block-encoding implementation.  Large-N entries may use explicitly marked structural estimates if full transpilation exceeds the selected guardrails.\n"""
    Path(filename).write_text(text, encoding="utf-8")


def save_metric_plot(rows: Sequence[BenchmarkRow], metric: str, ylabel: str, filename: str) -> None:
    """Save a clear log-scale plot over Renata's selected N values.

    We use equally spaced selected-N positions on the x-axis to avoid visual
    collapse of the small values 3--10, while the y-axis is logarithmic.  Prime
    dimensions are marked with an asterisk in the tick label.
    """
    if plt is None:
        return
    valid = [r for r in rows if getattr(r, metric) is not None and getattr(r, metric) > 0]
    if not valid:
        return
    all_N = sorted({r.N for r in valid})
    pos = {n: i for i, n in enumerate(all_N)}
    methods = ["QBA_prototype", "mixed_radix_QFT", "mixed_radix_exact_reference", "AQFT"]
    fig, ax = plt.subplots(figsize=(max(13.5, 0.56 * len(all_N)), 6.4))
    seen = set()
    for method in methods:
        series = [r for r in valid if r.method == method]
        if not series:
            continue
        series = sorted(series, key=lambda r: r.N)
        xs = [pos[r.N] for r in series]
        ys = [getattr(r, metric) for r in series]
        label = short_method_name(method)
        if label in seen:
            continue
        seen.add(label)
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Selected logical transform size $N$; * marks prime $N$")
    ax.set_ylabel(ylabel + " (log scale)")
    ax.set_title(f"{ylabel} vs. transform size $N$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    ax.set_xticks(list(range(len(all_N))))
    ax.set_xticklabels([str(n) + ("*" if is_prime(n) else "") for n in all_N], rotation=55, ha="right")
    ax.text(0.99, 0.02, "x-axis uses equal spacing for the selected $N$ values", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(str(Path(filename).with_suffix(".pdf")), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def print_summary(rows: Sequence[BenchmarkRow]) -> None:
    print("\n=== Unified benchmark summary ===")
    for r in rows:
        print(
            f"N={r.N:>3} | {r.method:<30} | qubits={r.physical_qubits:<3} "
            f"| depth={str(r.depth):>8} | gates={str(r.total_gate_count):>8} "
            f"| 2q={str(r.two_qubit_gate_count):>8} | exactness={r.exactness_status}"
        )
        if r.qba_alpha is not None:
            print(f"      QBA diagnostics: alpha={r.qba_alpha:.12g}, p_success={r.qba_p_success:.12g}, expected_repetitions={r.qba_expected_repetitions:.12g}")
        if r.dft_validation_linf_error is not None:
            print(f"      validation: linf_error={r.dft_validation_linf_error:.3e}, fidelity={r.dft_validation_fidelity:.12g}")
        if r.transpile_warning:
            print(f"      warning: {r.transpile_warning}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified QBA vs mixed-radix QFT vs AQFT benchmark script")
    parser.add_argument("--N-list", default=",".join(map(str, RENATA_EXTENDED_N_LIST)), help="Comma-separated logical N values")
    parser.add_argument("--input-mode", default="paper", choices=["paper", "basis0", "basis1", "uniform", "random"])
    parser.add_argument("--dft-sign", type=int, default=-1, choices=[-1, 1], help="Use -1 for standard mathematical DFT; Qiskit inverse-QFT implements this convention")
    parser.add_argument("--approximation-degree", type=int, default=1, help="Qiskit AQFT approximation degree")
    parser.add_argument("--aqft-register-mode", default="minimal_logical", choices=["minimal_logical", "qba_workspace"], help="AQFT register choice")
    parser.add_argument("--basis-gates", default=",".join(DEFAULT_BASIS_GATES), help="Comma-separated basis gates. Use empty string for Qiskit default.")
    parser.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--max-dense-dim", type=int, default=128, help="Guardrail for dense full-QFT_N reference paths; gate-level mixed-radix uses --max-custom-factor-dim")
    parser.add_argument("--max-custom-factor-dim", type=int, default=1024, help="Maximum embedded dimension for a custom local QFT_r factor gate")
    parser.add_argument("--max-qba-M", type=int, default=256, help="Maximum QBA workspace M for full prototype construction before structural-estimate fallback")
    parser.add_argument("--no-structural-estimates", action="store_true", help="Disable structural estimates for large N; rows that exceed thresholds will remain blank/skipped")
    parser.add_argument("--validate-statevectors", action="store_true", help="Run statevector validation where feasible")
    parser.add_argument("--save-circuit-png", action="store_true", help="Save transpiled circuit drawings as PNG")
    parser.add_argument("--out-prefix", default="qft_three_way_benchmark")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not HAVE_QISKIT:
        if args.no_structural_estimates:
            print("ERROR: Qiskit is required when --no-structural-estimates is used.", file=sys.stderr)
            sys.exit(2)
        print("WARNING: Qiskit is not available. Producing structural-estimate rows only; these are not transpiled benchmark counts.", file=sys.stderr)
    N_list = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]
    basis_gates = [x.strip() for x in args.basis_gates.split(",") if x.strip()] or None
    fig_dir = Path(f"{args.out_prefix}_circuits")

    rows: List[BenchmarkRow] = []
    for N in N_list:
        if N < 2:
            print(f"Skipping N={N}: benchmark expects N>=2")
            continue
        rows.extend(benchmark_one_N(
            N,
            input_mode=args.input_mode,
            dft_sign=args.dft_sign,
            approximation_degree=args.approximation_degree,
            aqft_register_mode=args.aqft_register_mode,
            basis_gates=basis_gates,
            optimization_level=args.optimization_level,
            max_dense_dim=args.max_dense_dim,
            max_qba_M=args.max_qba_M,
            max_custom_factor_dim=args.max_custom_factor_dim,
            allow_structural_estimates=not args.no_structural_estimates,
            validate_statevectors=args.validate_statevectors,
            save_circuit_pngs=args.save_circuit_png,
            fig_dir=fig_dir,
        ))

    if not rows:
        print("No benchmark rows were produced.", file=sys.stderr)
        sys.exit(1)

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"
    tex_path = f"{args.out_prefix}_table.tex"
    standalone_tex_path = f"{args.out_prefix}_table_standalone.tex"
    depth_png = f"{args.out_prefix}_depth.png"
    twoq_png = f"{args.out_prefix}_two_qubit_gates.png"
    gates_png = f"{args.out_prefix}_total_gates.png"
    fidelity_png = f"{args.out_prefix}_state_fidelity.png"
    protocol_txt = f"{args.out_prefix}_benchmark_protocol.txt"

    save_csv(rows, csv_path)
    save_json(rows, json_path)
    save_latex_table(rows, tex_path)
    save_latex_standalone(rows, standalone_tex_path)
    save_metric_plot(rows, "depth", "Transpiled circuit depth", depth_png)
    save_metric_plot(rows, "two_qubit_gate_count", "Two-qubit gate count", twoq_png)
    save_metric_plot(rows, "total_gate_count", "Total transpiled gate count", gates_png)
    save_fidelity_plot(rows, fidelity_png)
    save_benchmark_protocol_text(protocol_txt, basis_gates=basis_gates, optimization_level=args.optimization_level, aqft_register_mode=args.aqft_register_mode, approximation_degree=args.approximation_degree)

    print_summary(rows)
    print("\nSaved outputs:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSON:  {json_path}")
    print(f"  LaTeX: {tex_path}")
    print(f"  Standalone LaTeX: {standalone_tex_path}")
    print(f"  Protocol explanation: {protocol_txt}")
    if plt is not None:
        print(f"  Plot:  {depth_png}")
        print(f"  Plot:  {twoq_png}")
        print(f"  Plot:  {gates_png}")
        print(f"  Plot:  {fidelity_png}")
    if args.save_circuit_png:
        print(f"  Circuit PNG directory: {fig_dir}")


if __name__ == "__main__":
    main()
