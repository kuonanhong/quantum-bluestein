#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Renata Wong
# Copyright (c) 2026 Nan-Hong Kuo
#
# This file is a modified version of the N=6 Quantum Bluestein worked-example
# code originally developed in Renata Wong's quantum-bluestein repository:
# https://github.com/renatawong/quantum-bluestein
#
# Original source basis:
#   quantum-bluestein-n6.ipynb
#
# Modifications by Nan-Hong Kuo include:
#   - conversion to a pure Python executable script;
#   - automatic circuit and histogram figure generation;
#   - paper-aligned input state for the Appendix A N=6 worked example;
#   - post-selected logical-output analysis;
#   - resource, benchmark, and validation diagnostics.
#
# Distributed under the MIT License.
"""
General revised proof-of-concept QBA script.
Worked-example aligned version for Appendix A, N=6.
(Fixed for pure Python execution: Automatically generates and saves figures as .png)
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import DiagonalGate, QFT
    try:
        from qiskit.circuit.library import StatePreparation
    except Exception:
        StatePreparation = None
    try:
        from qiskit.circuit.library import UnitaryGate
    except Exception:
        try:
            from qiskit.extensions import UnitaryGate
        except Exception:
            UnitaryGate = None
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError:
        QiskitRuntimeService = None
        Sampler = None
    HAVE_QISKIT = True
except ImportError:
    ClassicalRegister = None
    QuantumCircuit = None
    QuantumRegister = None
    DiagonalGate = None
    QFT = None
    Statevector = None
    QiskitRuntimeService = None
    Sampler = None
    StatePreparation = None
    UnitaryGate = None
    transpile = None
    plot_histogram = None
    HAVE_QISKIT = False

DEFAULT_N = 6
DEFAULT_FIXED_INPUT = [1, 1, 1, 0, 0, 0]
DEFAULT_SEED = 7
DEFAULT_SHOTS = 10000
DEFAULT_APPROX_DEGREE = 1


def choose_workspace_size(N: int, workspace_mode: str = "power_of_two") -> tuple[int, int]:
    min_size = 2 * N - 1
    if workspace_mode == "power_of_two":
        m = int(np.ceil(np.log2(min_size)))
        M = 2 ** m
        return m, M
    if workspace_mode == "exact_minimum":
        M = min_size
        m = int(np.ceil(np.log2(M)))
        return m, M
    if workspace_mode == "next_smooth_length":
        M = min_size
        while M % 2 != 0 and M < min_size + 32:
            M += 1
        m = int(np.ceil(np.log2(M)))
        return m, M
    raise ValueError(f"Unknown workspace_mode={workspace_mode!r}")


@dataclass
class ClassicalProfile:
    N: int
    M: int
    m: int
    workspace_mode: str
    x_in: np.ndarray
    padded_input: np.ndarray
    chirp_in: np.ndarray
    chirp_out: np.ndarray
    b_vec: np.ndarray
    b_tilde: np.ndarray
    alpha: float
    psi2: np.ndarray
    useful_branch_full: np.ndarray
    useful_branch_logical: np.ndarray
    target_n_dft: np.ndarray
    p_ancilla_success: float
    p_logical_unconditional: float
    p_logical_given_ancilla: float
    expected_repetitions: float
    aa_repetitions: float


@dataclass
class NoiseSummary:
    label: str
    fidelity_to_target: float
    logical_hist: Dict[str, float]
    note: str


def normalize_state(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Input vector must not be the zero vector.")
    return vec / norm


def choose_input_vector(N: int, fixed_input: Optional[Iterable[complex]], seed: int) -> np.ndarray:
    if fixed_input is not None:
        arr = np.asarray(list(fixed_input), dtype=complex)
        if arr.shape[0] != N:
            raise ValueError(f"Fixed input length {arr.shape[0]} does not match N={N}.")
        return normalize_state(arr)
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=N) + 1j * rng.normal(size=N)
    return normalize_state(arr)


def quantize_angle(theta: float, angle_precision: Optional[float]) -> float:
    if angle_precision is None:
        return float(theta)
    if angle_precision <= 0:
        raise ValueError("angle_precision must be positive when provided.")
    return float(np.round(theta / angle_precision) * angle_precision)


def bitstrings(width: int):
    for i in range(2 ** width):
        yield format(i, f"0{width}b")


def two_qubit_gate_count_from_ops(op_counts: Dict[str, int]) -> int:
    two_qubit_names = {"cx", "cz", "cp", "swap", "ecr", "rzz", "rxx", "ryy", "crz", "cry", "crx", "cswap"}
    return int(sum(count for name, count in op_counts.items() if name in two_qubit_names))



def save_figure_tight(fig: Any, stem: str, *, dpi: int = 600, pad_inches: float = 0.02, save_pdf: bool = True, save_svg: bool = False) -> Dict[str, str]:
    """Save a matplotlib figure with minimal surrounding whitespace.

    This is intended for journal-ready export of circuit figures when quantikz
    is not supported by the submission system.
    """
    out: Dict[str, str] = {}
    stem_path = Path(stem)
    png_path = stem_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches, facecolor='white')
    out['png'] = str(png_path)
    if save_pdf:
        pdf_path = stem_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches=pad_inches, facecolor='white')
        out['pdf'] = str(pdf_path)
    if save_svg:
        svg_path = stem_path.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches='tight', pad_inches=pad_inches, facecolor='white')
        out['svg'] = str(svg_path)
    return out


def save_circuit_mpl_tight(circuit: Any, stem: str, *, scale: float = 0.6, fold: int = -1, dpi: int = 600, pad_inches: float = 0.02) -> Dict[str, str]:
    fig = circuit.draw(output='mpl', scale=scale, fold=fold)
    try:
        fig.tight_layout(pad=0.05)
    except Exception:
        pass
    saved = save_figure_tight(fig, stem, dpi=dpi, pad_inches=pad_inches, save_pdf=True, save_svg=False)
    if plt is not None:
        plt.close(fig)
    return saved



def factor_integer(n: int) -> list[int]:
    """Return the prime factors of n. Used only for reviewer-facing benchmark metadata."""
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def dft_matrix(n: int, sign: int = -1) -> np.ndarray:
    """Normalized n-point DFT matrix."""
    j = np.arange(n)
    k = j[:, None]
    return np.exp(sign * 2j * np.pi * k * j / n) / np.sqrt(n)


def embedded_qft_unitary(logical_dim: int, physical_dim: int, sign: int = -1) -> np.ndarray:
    """Block-diagonal unitary: exact QFT on the first logical_dim states, identity elsewhere.

    This is a dense exact-reference construction for controlled small-N comparisons.
    It is not advertised as an optimized mixed-radix decomposition.
    """
    if logical_dim > physical_dim:
        raise ValueError("logical_dim must not exceed physical_dim")
    U = np.eye(physical_dim, dtype=complex)
    U[:logical_dim, :logical_dim] = dft_matrix(logical_dim, sign=sign)
    return U


def build_exact_qft_dense_reference(profile: "ClassicalProfile"):
    """Build a dense embedded exact-QFT_N reference circuit for small-N benchmarking."""
    if not HAVE_QISKIT or UnitaryGate is None:
        return None
    physical_dim = 2 ** profile.m
    # Avoid accidental gigantic dense synthesis in exploratory runs.
    if physical_dim > 64:
        return None
    qc = QuantumCircuit(profile.m, name=f"exact_QFT_{profile.N}_dense_ref")
    U = embedded_qft_unitary(profile.N, physical_dim, sign=-1)
    qc.append(UnitaryGate(U, label=f"ExactQFT_N{profile.N}_dense"), range(profile.m))
    return qc


def save_logical_histogram(hist: Dict[str, float], N: int, filename: str, title: Optional[str] = None) -> None:
    """Save a publication-facing logical histogram without Qiskit's quasi-probability label."""
    keys = list(hist.keys())
    vals = [hist[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(keys, vals)
    ax.set_xlabel(r"Logical output basis state $k$")
    ax.set_ylabel("Post-selected logical probability")
    ax.set_title(title or f"Bluestein QFT logical output (N={N})")
    ax.set_ylim(0, max(vals) * 1.18 if vals else 1.0)
    fig.tight_layout()
    save_figure_tight(fig, Path(filename).with_suffix(''), dpi=600, pad_inches=0.02, save_pdf=True); filename = str(Path(filename).with_suffix('.png'))
    plt.close(fig)


def json_safe(obj: Any) -> Any:
    """Convert numpy/complex objects into JSON-safe objects."""
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [{"real": complex(x).real, "imag": complex(x).imag} for x in obj.ravel().tolist()]
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj


def write_metrics_json(filename: str, payload: Dict[str, Any]) -> None:
    Path(filename).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def apply_success_reflection(qc: "QuantumCircuit", anc: "QuantumRegister") -> None:
    """Phase flip on the success subspace ancilla=|0>."""
    qc.x(anc[0])
    qc.z(anc[0])
    qc.x(anc[0])


def apply_zero_reflection(qc: "QuantumCircuit", qubits: list) -> None:
    """Phase flip on the all-zero state of all supplied qubits."""
    for qb in qubits:
        qc.x(qb)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for qb in qubits:
        qc.x(qb)


def loader_rotation_count(M: int) -> int:
    return int(M)


def report_naive_loader_cost(profile: ClassicalProfile) -> Dict[str, Any]:
    return {
        "naive_loader_state_conditioned_rotations": loader_rotation_count(profile.M),
        "basis_labels_loaded": profile.M,
        "comment": (
            "Naive basis-by-basis diagonal loading for demonstration only; "
            "not a structured low-cost oracle for arbitrary tilde b_k."
        ),
    }


def build_classical_profile(
    N: int,
    fixed_input: Optional[Iterable[complex]] = None,
    seed: int = DEFAULT_SEED,
    workspace_mode: str = "power_of_two",
) -> ClassicalProfile:
    if N <= 0:
        raise ValueError("N must be positive.")
    min_size = 2 * N - 1
    m, M = choose_workspace_size(N, workspace_mode=workspace_mode)
    x_in = choose_input_vector(N, fixed_input, seed)
    padded_input = np.zeros(M, dtype=complex)
    padded_input[:N] = x_in

    idx = np.arange(M)
    chirp_in = np.exp(-1j * np.pi * (idx ** 2) / N)
    chirp_out = np.exp(-1j * np.pi * (idx ** 2) / N)

    b_vec = np.zeros(M, dtype=complex)
    for t in range(N):
        val = np.exp(1j * np.pi * (t ** 2) / N)
        b_vec[t] = val
        if t > 0:
            b_vec[M - t] = val

    b_tilde = np.fft.fft(b_vec)
    alpha = float(np.max(np.abs(b_tilde)))

    a = padded_input * chirp_in
    psi2 = np.fft.fft(a) / np.sqrt(M)
    fourier_multiplied = psi2 * (b_tilde / alpha)
    useful_branch_full = (np.sqrt(M) * np.fft.ifft(fourier_multiplied)) * chirp_out
    useful_branch_logical = useful_branch_full[:N].copy()
    target_n_dft = np.fft.fft(x_in) / np.sqrt(N)

    p_ancilla_success = float(np.sum(np.abs(psi2) ** 2 * (np.abs(b_tilde) ** 2) / (alpha ** 2)).real)
    p_logical_unconditional = float(np.sum(np.abs(useful_branch_logical) ** 2).real)
    p_logical_given_ancilla = p_logical_unconditional / p_ancilla_success if p_ancilla_success > 0 else 0.0
    expected_repetitions = math.inf if p_ancilla_success <= 0 else 1.0 / p_ancilla_success
    aa_repetitions = math.inf if p_ancilla_success <= 0 else 1.0 / math.sqrt(p_ancilla_success)

    return ClassicalProfile(
        N=N, M=M, m=m, workspace_mode=workspace_mode, x_in=x_in, padded_input=padded_input,
        chirp_in=chirp_in, chirp_out=chirp_out, b_vec=b_vec, b_tilde=b_tilde,
        alpha=alpha, psi2=psi2, useful_branch_full=useful_branch_full,
        useful_branch_logical=useful_branch_logical, target_n_dft=target_n_dft,
        p_ancilla_success=p_ancilla_success,
        p_logical_unconditional=p_logical_unconditional,
        p_logical_given_ancilla=p_logical_given_ancilla,
        expected_repetitions=expected_repetitions,
        aa_repetitions=aa_repetitions,
    )


def apply_fft(qc: "QuantumCircuit", qubits: list) -> None:
    n = len(qubits)
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - 1 - i])
    for i in range(n):
        qc.h(qubits[i])
        for j in range(i + 1, n):
            qc.cp(-np.pi / (2 ** (j - i)), qubits[j], qubits[i])


def apply_ifft(qc: "QuantumCircuit", qubits: list) -> None:
    n = len(qubits)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            qc.cp(np.pi / (2 ** (j - i)), qubits[j], qubits[i])
        qc.h(qubits[i])
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - 1 - i])


def apply_structured_quadratic_phase(
    qc: "QuantumCircuit",
    qubits: list,
    N: int,
    sign: float,
    angle_precision: Optional[float] = None,
    angle_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    n = len(qubits)
    rng = np.random.default_rng() if rng is None else rng
    for r in range(n):
        theta = sign * np.pi * (2 ** (2 * r)) / N
        if angle_noise_std > 0:
            theta += rng.normal(scale=angle_noise_std)
        qc.p(quantize_angle(theta, angle_precision), qubits[r])
        for s in range(r + 1, n):
            theta = sign * 2.0 * np.pi * (2 ** (r + s)) / N
            if angle_noise_std > 0:
                theta += rng.normal(scale=angle_noise_std)
            qc.cp(quantize_angle(theta, angle_precision), qubits[s], qubits[r])


def apply_conv_diagonal_naive(
    qc: "QuantumCircuit",
    qr: "QuantumRegister",
    anc: "QuantumRegister",
    normalized_b_tilde: np.ndarray,
    angle_precision: Optional[float] = None,
    angle_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng() if rng is None else rng
    phase_angles = np.angle(normalized_b_tilde)
    if angle_noise_std > 0:
        phase_angles = phase_angles + rng.normal(scale=angle_noise_std, size=len(phase_angles))
    qc.append(DiagonalGate(np.exp(1j * np.array([quantize_angle(a, angle_precision) for a in phase_angles]))), qr)

    magnitudes = np.abs(normalized_b_tilde)
    thetas = 2.0 * np.arccos(np.clip(magnitudes, 0.0, 1.0))
    appended = 0
    for k, theta in enumerate(thetas):
        if angle_noise_std > 0:
            theta += rng.normal(scale=angle_noise_std)
        theta = quantize_angle(float(theta), angle_precision)
        if abs(theta) <= 1e-12:
            continue
        sub = QuantumCircuit(1)
        sub.ry(theta, 0)
        ctrl_gate = sub.to_gate(label=f"RY_k{k}").control(len(qr), ctrl_state=k)
        qc.append(ctrl_gate, list(qr) + [anc[0]])
        appended += 1
    return {
        "state_conditioned_rotations_appended": appended,
        "basis_size_loaded": len(normalized_b_tilde),
    }


def build_qba_circuit(
    profile: ClassicalProfile,
    angle_precision: Optional[float] = None,
    use_structured_chirp: bool = False,
    add_measurements: bool = False,
    angle_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    stateprep_mode: str = "initialize",
):
    if not HAVE_QISKIT:
        raise ImportError("qiskit is not installed. Only classical metrics can be computed in this environment.")
    rng = np.random.default_rng() if rng is None else rng
    qr = QuantumRegister(profile.m, "q")
    anc = QuantumRegister(1, "anc")
    if add_measurements:
        cr = ClassicalRegister(profile.m + 1, "c")
        qc = QuantumCircuit(qr, anc, cr)
    else:
        qc = QuantumCircuit(qr, anc)

    if stateprep_mode == "unitary":
        if StatePreparation is None:
            raise RuntimeError("StatePreparation is unavailable; cannot build an invertible state-preparation block for amplitude amplification.")
        qc.append(StatePreparation(profile.padded_input), qr)
    else:
        qc.initialize(profile.padded_input, qr)
    if use_structured_chirp:
        apply_structured_quadratic_phase(qc, list(qr), profile.N, sign=-1.0,
                                         angle_precision=angle_precision,
                                         angle_noise_std=angle_noise_std, rng=rng)
    else:
        qc.append(DiagonalGate(profile.chirp_in), qr)

    apply_fft(qc, list(qr))
    loader_info = apply_conv_diagonal_naive(
        qc, qr, anc, profile.b_tilde / profile.alpha,
        angle_precision=angle_precision,
        angle_noise_std=angle_noise_std, rng=rng,
    )
    apply_ifft(qc, list(qr))

    if use_structured_chirp:
        apply_structured_quadratic_phase(qc, list(qr), profile.N, sign=-1.0,
                                         angle_precision=angle_precision,
                                         angle_noise_std=angle_noise_std, rng=rng)
    else:
        qc.append(DiagonalGate(profile.chirp_out), qr)

    if add_measurements:
        qc.measure(anc, qc.clbits[profile.m])
        qc.measure(qr, qc.clbits[:profile.m])

    resource_report = {
        "explicit_ancillas": 1,
        "optional_work_qubits": "compilation-strategy dependent",
        "workspace_qubits": profile.m,
        "workspace_dimension_M": profile.M,
        "workspace_mode": profile.workspace_mode,
        "logical_dimension_N": profile.N,
        "uses_generic_diagonal_placeholder_for_chirp": not use_structured_chirp,
        "uses_naive_diagonal_loader": True,
        "convolution_model": "ancilla-assisted block encoding with naive basis-conditioned Ry loading",
        "naive_loader_state_conditioned_rotations": loader_info["state_conditioned_rotations_appended"],
        "optimized_mixed_radix_qft_implemented": False,
        "dense_exact_qft_reference_available_for_small_N": bool(UnitaryGate is not None and 2 ** profile.m <= 64),
        "amplitude_amplification_circuit_available": bool(StatePreparation is not None),
        "alpha": profile.alpha,
        "note": (
            "This example uses 1 explicit ancilla, but structured or QROM-based compilation "
            "may require additional work qubits. The current circuit path is implemented only "
            "for workspace_mode=power_of_two, reflecting the radix-2 design choice rather than "
            "the mathematical alias-free lower bound alone."
        ),
    }
    return qc, resource_report


def build_amplitude_amplification_circuit(
    profile: ClassicalProfile,
    iterations: int = 1,
    angle_precision: Optional[float] = None,
    use_structured_chirp: bool = False,
):
    """Build an optional amplitude-amplification demonstration circuit.

    This is intentionally off by default.  It exists to make clear what would be
    required if the manuscript discusses amplitude amplification.  The resource
    cost of this circuit must be counted separately.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not HAVE_QISKIT:
        raise ImportError("qiskit is not installed")
    A, _ = build_qba_circuit(
        profile,
        angle_precision=angle_precision,
        use_structured_chirp=use_structured_chirp,
        add_measurements=False,
        stateprep_mode="unitary",
    )
    A_gate = A.to_gate(label="A_QBA")
    qr = QuantumRegister(profile.m, "q")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(qr, anc, name=f"QBA_AA_N{profile.N}")
    all_qubits = list(qr) + list(anc)
    qc.append(A_gate, all_qubits)
    for _ in range(iterations):
        apply_success_reflection(qc, anc)
        qc.append(A_gate.inverse(), all_qubits)
        apply_zero_reflection(qc, all_qubits)
        qc.append(A_gate, all_qubits)
    return qc


def amplitude_amplification_resource_note(profile: ClassicalProfile, iterations: int) -> Dict[str, Any]:
    return {
        "implemented_as_optional_demo": True,
        "enabled_by_default": False,
        "iterations_requested": int(iterations),
        "baseline_success_probability": profile.p_ancilla_success,
        "baseline_expected_repetitions": profile.expected_repetitions,
        "warning": (
            "Amplitude amplification is not part of the default QBA prototype. "
            "If discussed in the manuscript, its additional circuit depth and oracle/reflection cost must be included."
        ),
    }


def align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(reference)))
    if np.abs(reference[idx]) <= 1e-14 or np.abs(candidate[idx]) <= 1e-14:
        return candidate
    return candidate * np.exp(1j * (np.angle(reference[idx]) - np.angle(candidate[idx])))


def analyze_statevector_result(profile: ClassicalProfile, statevector: np.ndarray) -> Dict[str, Any]:
    raw_data = np.asarray(statevector, dtype=complex)
    fmt = f"0{profile.m + 1}b"
    ancilla_zero = np.zeros(profile.M, dtype=complex)
    ancilla_one_mass = 0.0
    for i, amp in enumerate(raw_data):
        bitstr = format(i, fmt)
        if bitstr[0] == "0":
            ancilla_zero[int(bitstr[1:], 2)] = amp
        else:
            ancilla_one_mass += float(np.abs(amp) ** 2)

    logical_branch = ancilla_zero[:profile.N]
    garbage_branch = ancilla_zero[profile.N:]
    p_ancilla_success = float(np.sum(np.abs(ancilla_zero) ** 2).real)
    p_logical_unconditional = float(np.sum(np.abs(logical_branch) ** 2).real)
    p_garbage_unconditional = float(np.sum(np.abs(garbage_branch) ** 2).real)
    p_logical_given_ancilla = p_logical_unconditional / p_ancilla_success if p_ancilla_success > 0 else 0.0

    target = profile.target_n_dft
    predicted_scaled = target / profile.alpha
    q_norm = logical_branch / np.linalg.norm(logical_branch) if np.linalg.norm(logical_branch) > 0 else logical_branch
    target_norm = target / np.linalg.norm(target) if np.linalg.norm(target) > 0 else target
    q_aligned = align_global_phase(target_norm, q_norm)
    scaled_aligned = align_global_phase(predicted_scaled, logical_branch)
    ls_scale = complex(np.vdot(target, logical_branch) / np.vdot(target, target)) if np.vdot(target, target) != 0 else 0.0j

    return {
        "p_ancilla_success": p_ancilla_success,
        "p_logical_unconditional": p_logical_unconditional,
        "p_garbage_unconditional": p_garbage_unconditional,
        "p_logical_given_ancilla": p_logical_given_ancilla,
        "p_total": p_logical_unconditional,
        "expected_repetitions": math.inf if p_ancilla_success <= 0 else 1.0 / p_ancilla_success,
        "aa_repetitions": math.inf if p_ancilla_success <= 0 else 1.0 / math.sqrt(p_ancilla_success),
        "unnormalized_success_branch_norm": float(np.linalg.norm(ancilla_zero)),
        "unnormalized_logical_branch_norm": float(np.linalg.norm(logical_branch)),
        "logical_branch": logical_branch,
        "target_n_dft": target,
        "predicted_scaled_target": predicted_scaled,
        "least_squares_scale": ls_scale,
        "expected_scale_1_over_alpha": 1.0 / profile.alpha,
        "normalized_max_error_after_projection": float(np.max(np.abs(q_aligned - target_norm))) if target_norm.size else 0.0,
        "unnormalized_max_error_vs_target_over_alpha": float(np.max(np.abs(scaled_aligned - predicted_scaled))) if predicted_scaled.size else 0.0,
        "ancilla_one_probability": ancilla_one_mass,
    }


def analyze_hardware_counts(profile: ClassicalProfile, counts: Dict[str, int]) -> Dict[str, Any]:
    total_shots = int(sum(counts.values()))
    success_shots = 0
    useful_shots = 0
    ancilla_zero_counts: Dict[str, int] = {}
    logical_counts: Dict[str, int] = {}
    for bitstring, count in counts.items():
        ancilla_state = bitstring[0]
        if ancilla_state == "0":
            success_shots += count
            data_part = bitstring[1:]
            ancilla_zero_counts[data_part] = ancilla_zero_counts.get(data_part, 0) + count
            idx = int(data_part, 2)
            if idx < profile.N:
                useful_shots += count
                logical_counts[data_part] = logical_counts.get(data_part, 0) + count
    p_ancilla = success_shots / total_shots if total_shots > 0 else 0.0
    p_total_useful = useful_shots / total_shots if total_shots > 0 else 0.0
    logical_given_ancilla = useful_shots / success_shots if success_shots > 0 else 0.0
    return {
        "total_shots": total_shots,
        "success_shots": success_shots,
        "useful_shots": useful_shots,
        "p_ancilla_success": p_ancilla,
        "p_total_useful": p_total_useful,
        "p_total": p_total_useful,
        "p_logical_given_ancilla": logical_given_ancilla,
        "expected_repetitions": math.inf if p_ancilla <= 0 else 1.0 / p_ancilla,
        "aa_repetitions": math.inf if p_ancilla <= 0 else 1.0 / math.sqrt(p_ancilla),
        "ancilla_zero_counts": ancilla_zero_counts,
        "logical_counts": logical_counts,
    }


def benchmark_circuit_set(profile: ClassicalProfile, qba_circuit: "QuantumCircuit", approximation_degree: int = DEFAULT_APPROX_DEGREE) -> Dict[str, Dict[str, Any]]:
    if not HAVE_QISKIT:
        return {}
    circuits = {"qba_prototype": qba_circuit}
    exact_ref = build_exact_qft_dense_reference(profile)
    if exact_ref is not None:
        circuits["exact_qft_N_dense_reference"] = exact_ref
    try:
        zero_pad = QuantumCircuit(profile.m)
        zero_pad.append(QFT(profile.m, approximation_degree=0, do_swaps=True).decompose(), range(profile.m))
        circuits["zero_padding_qft_M"] = zero_pad
        aqft = QuantumCircuit(profile.m)
        aqft.append(QFT(profile.m, approximation_degree=approximation_degree, do_swaps=True).decompose(), range(profile.m))
        circuits[f"aqft_deg_{approximation_degree}"] = aqft
    except Exception:
        pass
    out = {}
    for name, circ in circuits.items():
        tcirc = transpile(circ, optimization_level=3) if transpile is not None else circ
        op_counts = {str(k): int(v) for k, v in tcirc.count_ops().items()}
        out[name] = {
            "depth": int(tcirc.depth()),
            "size": int(tcirc.size()),
            "width": int(tcirc.num_qubits),
            "count_ops": op_counts,
            "two_qubit_gates": two_qubit_gate_count_from_ops(op_counts),
        }
    out["benchmark_limitations"] = {
        "optimized_mixed_radix_exact_qft": "not implemented",
        "dense_exact_qft_reference": "included for small N when UnitaryGate is available; not an optimized mixed-radix decomposition",
        "factorization_of_N": factor_integer(profile.N),
        "methodology_note": "QBA, zero-padding, AQFT, and the dense exact reference are transpiled under the same local Qiskit settings when available.",
    }
    return out


def logical_hist_from_state(logical_state: np.ndarray, threshold: float = 1e-10) -> Dict[str, float]:
    """Return only non-negligible logical probabilities for paper worked examples.

    This avoids displaying tiny numerical roundoff bars for states whose
    theoretical probability is exactly zero, e.g. states 2 and 4 in the
    Appendix A N=6 example.
    """
    N = len(logical_state)
    m = int(np.ceil(np.log2(max(1, N))))
    probs = np.abs(logical_state) ** 2
    total = float(np.sum(probs))
    if total > 0:
        probs = probs / total
    return {format(i, f"0{m}b"): float(probs[i]) for i in range(N) if float(probs[i]) > threshold}


def histogram_fidelity(h1: Dict[str, float], h2: Dict[str, float]) -> float:
    keys = sorted(set(h1) | set(h2))
    return float(sum(np.sqrt(h1.get(k, 0.0) * h2.get(k, 0.0)) for k in keys) ** 2)


def apply_readout_error(hist: Dict[str, float], width: int, readout_p: float) -> Dict[str, float]:
    if readout_p <= 0:
        return dict(hist)
    out = {k: 0.0 for k in hist.keys()}
    all_keys = list(hist.keys())
    for src, psrc in hist.items():
        bits_src = np.array([int(b) for b in src], dtype=int)
        for dst in all_keys:
            bits_dst = np.array([int(b) for b in dst], dtype=int)
            flips = np.sum(bits_src != bits_dst)
            same = width - flips
            p = (readout_p ** flips) * ((1.0 - readout_p) ** same)
            out[dst] += psrc * p
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def surrogate_noisy_branch(profile: ClassicalProfile, angle_noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(profile.M)

    in_angles = -np.pi * (idx ** 2) / profile.N + rng.normal(scale=angle_noise_std, size=profile.M)
    out_angles = -np.pi * (idx ** 2) / profile.N + rng.normal(scale=angle_noise_std, size=profile.M)
    noisy_chirp_in = np.exp(1j * in_angles)
    noisy_chirp_out = np.exp(1j * out_angles)

    phase_angles = np.angle(profile.b_tilde / profile.alpha) + rng.normal(scale=angle_noise_std, size=profile.M)
    phase_factor = np.exp(1j * phase_angles)
    theta_ideal = 2.0 * np.arccos(np.clip(np.abs(profile.b_tilde / profile.alpha), 0.0, 1.0))
    theta_noisy = theta_ideal + rng.normal(scale=angle_noise_std, size=profile.M)
    magnitudes_noisy = np.clip(np.cos(theta_noisy / 2.0), 0.0, 1.0)
    noisy_diag = phase_factor * magnitudes_noisy

    a = profile.padded_input * noisy_chirp_in
    psi2 = np.fft.fft(a) / np.sqrt(profile.M)
    branch = (np.sqrt(profile.M) * np.fft.ifft(psi2 * noisy_diag)) * noisy_chirp_out
    return branch[:profile.N].copy()


def build_noise_summaries(
    profile: ClassicalProfile,
    ideal_metrics: Dict[str, Any],
    benchmark_report: Dict[str, Dict[str, Any]],
    angle_noise_std: float,
    p_two_qubit: float,
    readout_p: float,
    seed: int,
) -> Dict[str, NoiseSummary]:
    target = profile.target_n_dft
    target_hist = logical_hist_from_state(target / np.linalg.norm(target))
    ideal_branch = ideal_metrics["logical_branch"]
    ideal_norm = ideal_branch / np.linalg.norm(ideal_branch) if np.linalg.norm(ideal_branch) > 0 else ideal_branch
    ideal_hist = logical_hist_from_state(ideal_norm)
    ideal_fid = float(np.abs(np.vdot(target / np.linalg.norm(target), ideal_norm)) ** 2) if np.linalg.norm(ideal_norm) > 0 else 0.0
    out = {
        "ideal": NoiseSummary("ideal", ideal_fid, ideal_hist, "Ideal post-selected logical branch."),
    }

    if angle_noise_std > 0:
        noisy_branch = surrogate_noisy_branch(profile, angle_noise_std=angle_noise_std, seed=seed + 101)
        noisy_norm = noisy_branch / np.linalg.norm(noisy_branch) if np.linalg.norm(noisy_branch) > 0 else noisy_branch
        fid = float(np.abs(np.vdot(target / np.linalg.norm(target), noisy_norm)) ** 2) if np.linalg.norm(noisy_norm) > 0 else 0.0
        out["angle_noise"] = NoiseSummary(
            "angle_noise",
            fid,
            logical_hist_from_state(noisy_norm),
            f"Simple surrogate coherent over-rotation model with std={angle_noise_std} rad.",
        )

    if p_two_qubit > 0:
        qba_name = "qba_prototype"
        n2 = benchmark_report.get(qba_name, {}).get("two_qubit_gates", profile.M)
        lam = 1.0 - (1.0 - p_two_qubit) ** max(1, int(n2))
        uniform = {k: 1.0 / len(ideal_hist) for k in ideal_hist.keys()}
        mixed_hist = {k: (1.0 - lam) * ideal_hist.get(k, 0.0) + lam * uniform[k] for k in ideal_hist.keys()}
        if readout_p > 0:
            width = len(next(iter(mixed_hist.keys()))) if mixed_hist else profile.m
            mixed_hist = apply_readout_error(mixed_hist, width, readout_p)
        fid = histogram_fidelity(target_hist, mixed_hist)
        out["two_qubit_noise"] = NoiseSummary(
            "two_qubit_noise",
            fid,
            mixed_hist,
            f"Simple surrogate depolarizing/readout model with p2={p_two_qubit}, readout_p={readout_p}, effective lambda={lam:.6f}.",
        )
    return out


def qml_exact_vs_zero_padded_demo(profile: ClassicalProfile) -> Dict[str, Any]:
    exact_feature = profile.target_n_dft
    zp_feature = np.fft.fft(profile.padded_input)[:profile.N] / np.sqrt(profile.M)
    cosine = float(np.abs(np.vdot(exact_feature, zp_feature)) / (np.linalg.norm(exact_feature) * np.linalg.norm(zp_feature)))
    return {
        "exact_feature": exact_feature,
        "zero_padded_feature_first_N": zp_feature,
        "feature_l2_difference": float(np.linalg.norm(exact_feature - zp_feature)),
        "cosine_similarity": cosine,
        "note": (
            "In kernel-based or physics-informed models, replacing the exact N-point spectrum by "
            "zero-padded M-point sampling can change the induced feature vector and therefore the kernel."
        ),
    }




def qml_gram_demo(N: int, samples: int = 4, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Small synthetic Gram-matrix diagnostic for exact vs zero-padded spectra.

    This supports only a modest claim: exact and zero-padded spectral embeddings
    can produce different feature vectors and kernel similarities.  It is not a
    classifier-margin, ECG, QCNN, or generalization experiment.
    """
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(samples):
        X.append(normalize_state(rng.normal(size=N) + 1j * rng.normal(size=N)))
    min_size = 2 * N - 1
    M = 2 ** int(np.ceil(np.log2(min_size)))
    exact_features = []
    padded_features = []
    for x in X:
        exact_features.append(np.fft.fft(x) / np.sqrt(N))
        pad = np.zeros(M, dtype=complex)
        pad[:N] = x
        padded_features.append((np.fft.fft(pad) / np.sqrt(M))[:N])
    K_exact = np.array([[abs(np.vdot(a, b)) ** 2 for b in exact_features] for a in exact_features])
    K_padded = np.array([[abs(np.vdot(a, b)) ** 2 for b in padded_features] for a in padded_features])
    return {
        "samples": samples,
        "N": N,
        "M_zero_padded": M,
        "gram_frobenius_difference": float(np.linalg.norm(K_exact - K_padded)),
        "max_entrywise_difference": float(np.max(np.abs(K_exact - K_padded))),
        "claim_supported": "Exact and zero-padded spectral embeddings can change kernel similarities on a synthetic example.",
        "claims_not_supported": [
            "classifier margin collapse",
            "predictive reliability degradation",
            "ECG/QCNN application effectiveness",
            "generalization improvement",
        ],
    }

def print_qml_demo(profile: ClassicalProfile) -> None:
    demo = qml_exact_vs_zero_padded_demo(profile)
    print("\n=== Classical exact-vs-zero-padded feature demo ===")
    print(f"feature_l2_difference = {demo['feature_l2_difference']:.12e}")
    print(f"cosine_similarity = {demo['cosine_similarity']:.12f}")
    print("exact_feature:")
    print(np.round(demo['exact_feature'], 6))
    print("zero_padded_feature_first_N:")
    print(np.round(demo['zero_padded_feature_first_N'], 6))
    print(demo['note'])
    gram = qml_gram_demo(profile.N, samples=4, seed=DEFAULT_SEED)
    print("gram_frobenius_difference =", f"{gram['gram_frobenius_difference']:.12e}")
    print("max_entrywise_difference =", f"{gram['max_entrywise_difference']:.12e}")
    print("QML diagnostic limitation:", "; ".join(gram["claims_not_supported"]))


def print_classical_report(profile: ClassicalProfile) -> None:
    print("\n=== Classical / analytic profile ===")
    print(f"N = {profile.N}, M = {profile.M}, m = {profile.m}")
    print(f"alpha = {profile.alpha:.12f}")
    print(f"Predicted ancilla success probability p_ancilla = {profile.p_ancilla_success:.12f}")
    print(f"Predicted logical-useful probability p_logical = {profile.p_logical_unconditional:.12f}")
    print(f"Predicted total useful probability p_total = p_ancilla * p_logical_given_ancilla = {profile.p_logical_unconditional:.12f}")
    print(f"Predicted logical retention given ancilla success = {profile.p_logical_given_ancilla:.12f}")
    print(f"Expected repetitions ~= 1/p = {profile.expected_repetitions:.6f}")
    print(f"Amplitude-amplification estimate 1/sqrt(p) = {profile.aa_repetitions:.6f} (resource estimate only; optional circuit requires --aa-demo)")
    print(f"Expected amplitude scale on ancilla-|0> branch = 1/alpha = {1.0/profile.alpha:.12f}")
    print(f"workspace_mode = {profile.workspace_mode}")
    print("Alias-free lower bound requires M >= 2N-1; choosing M = 2^m is the current radix-2 implementation choice.")


def print_statevector_report(metrics: Dict[str, Any]) -> None:
    print("\n=== Statevector-derived metrics ===")
    for key in [
        "p_ancilla_success", "p_logical_unconditional", "p_garbage_unconditional",
        "p_logical_given_ancilla", "expected_repetitions", "aa_repetitions",
        "p_total",
        "unnormalized_success_branch_norm", "unnormalized_logical_branch_norm",
        "expected_scale_1_over_alpha", "least_squares_scale",
        "normalized_max_error_after_projection", "unnormalized_max_error_vs_target_over_alpha",
        "ancilla_one_probability",
    ]:
        print(f"{key} = {metrics[key]}")
    print("target / alpha (first N amplitudes):")
    print(np.round(metrics["predicted_scaled_target"], 6))
    print("quantum logical branch (unnormalized):")
    print(np.round(metrics["logical_branch"], 6))


def print_benchmark_report(benchmarks: Dict[str, Dict[str, Any]]) -> None:
    if not benchmarks:
        print("\nBenchmark report unavailable.")
        return
    print("\n=== Prototype benchmark report ===")
    for name, rep in benchmarks.items():
        print(f"[{name}]")
        for k, v in rep.items():
            print(f"  {k}: {v}")


def print_noise_report(summaries: Dict[str, NoiseSummary]) -> None:
    print("\n=== Simple noise-model report ===")
    for key, summary in summaries.items():
        print(f"[{key}] fidelity_to_target = {summary.fidelity_to_target:.12f}")
        print(f"  note: {summary.note}")
        print(f"  logical_hist: {summary.logical_hist}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revised proof-of-concept Quantum Bluestein analysis script.")
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--simulation", action="store_true", default=True) # Always simulate by default
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--use-structured-chirp", action="store_true")
    parser.add_argument("--angle-precision", type=float, default=None)
    parser.add_argument("--angle-noise-std", type=float, default=0.0)
    parser.add_argument("--workspace-mode", choices=["power_of_two", "exact_minimum", "next_smooth_length"], default="power_of_two")
    parser.add_argument("--qml-demo", action="store_true")
    parser.add_argument("--compare-aqft", action="store_true", help="Force prototype benchmarking against zero-padding QFT_M and AQFT baselines.")
    parser.add_argument("--noise-model-mode", choices=["none", "simple"], default="none")
    parser.add_argument("--two-qubit-error", type=float, default=0.0)
    parser.add_argument("--readout-error", type=float, default=0.0)
    parser.add_argument("--approximation-degree", type=int, default=DEFAULT_APPROX_DEGREE)
    parser.add_argument("--aa-demo", action="store_true", help="Build an optional amplitude-amplification demonstration circuit and count its resources.")
    parser.add_argument("--aa-iterations", type=int, default=1)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_classical_profile(args.N, fixed_input=DEFAULT_FIXED_INPUT, seed=args.seed, workspace_mode=args.workspace_mode)
    print_classical_report(profile)
    print("\n=== Naive loader cost statement ===")
    for k, v in report_naive_loader_cost(profile).items():
        print(f"{k}: {v}")

    if not HAVE_QISKIT:
        print("\nQiskit not available in this environment: skipped circuit construction, simulation, and hardware execution.")
        return

    rng = np.random.default_rng(args.seed)
    qc, resource_report = build_qba_circuit(
        profile,
        angle_precision=args.angle_precision,
        use_structured_chirp=args.use_structured_chirp,
        add_measurements=args.hardware,
        angle_noise_std=args.angle_noise_std if args.noise_model_mode == "simple" else 0.0,
        rng=rng,
    )

    # Journal-ready circuit export (tight crop, high resolution, PNG + PDF)
    try:
        saved = save_circuit_mpl_tight(qc, f'qba_circuit_revised_N{args.N}', scale=0.6, fold=-1, dpi=600, pad_inches=0.02)
        print(f"\n[Visual] Circuit diagram saved as: {saved}")
    except Exception as e:
        print(f"\n[Visual] Could not draw or save circuit diagram: {e}")

    print("\n=== Resource statement ===")
    for k, v in resource_report.items():
        print(f"{k}: {v}")

    ideal_metrics = None
    if args.simulation:
        state = Statevector.from_instruction(qc)
        ideal_metrics = analyze_statevector_result(profile, np.asarray(state))
        print_statevector_report(ideal_metrics)
        
        # === FIXED: Save simulation histogram automatically ===
        try:
            norm_logical = ideal_metrics["logical_branch"] / np.linalg.norm(ideal_metrics["logical_branch"])
            hist_dict = logical_hist_from_state(norm_logical)
            sim_filename = f'qba_hist_logical_worked_N{args.N}.png'
            save_logical_histogram(
                hist_dict,
                args.N,
                sim_filename,
                title=f"Bluestein QFT worked example (N={args.N})",
            )
            print(f"\n[Visual] Worked-example logical histogram successfully saved as '{sim_filename}'")
        except Exception as e:
            print(f"\n[Visual] Could not save simulation histogram: {e}")

    benchmarks = {}
    if args.benchmark_mode or args.compare_aqft:
        benchmarks = benchmark_circuit_set(profile, qc, approximation_degree=args.approximation_degree)
        print_benchmark_report(benchmarks)
        if args.compare_aqft:
            print("\n=== Reviewer-facing AQFT comparison note ===")
            print("QBA and AQFT optimize different objectives: QBA targets transform exactness at arbitrary logical size, whereas AQFT sacrifices exactness to reduce rotation count and depth.")

    if args.aa_demo:
        print("\n=== Optional amplitude-amplification demo ===")
        try:
            aa_circ = build_amplitude_amplification_circuit(
                profile,
                iterations=args.aa_iterations,
                angle_precision=args.angle_precision,
                use_structured_chirp=args.use_structured_chirp,
            )
            taa = transpile(aa_circ, optimization_level=3) if transpile is not None else aa_circ
            aa_ops = {str(k): int(v) for k, v in taa.count_ops().items()}
            aa_report = amplitude_amplification_resource_note(profile, args.aa_iterations)
            aa_report.update({
                "aa_depth": int(taa.depth()),
                "aa_size": int(taa.size()),
                "aa_width": int(taa.num_qubits),
                "aa_count_ops": aa_ops,
                "aa_two_qubit_gates": two_qubit_gate_count_from_ops(aa_ops),
            })
            print(json.dumps(json_safe(aa_report), indent=2))
            try:
                saved_aa = save_circuit_mpl_tight(aa_circ, f"qba_amplitude_amplification_demo_N{args.N}", scale=0.45, fold=-1, dpi=600, pad_inches=0.02)
                print(f"[Visual] Amplitude-amplification demo circuit saved as: {saved_aa}")
            except Exception as e:
                print(f"[Visual] Could not save amplitude-amplification circuit: {e}")
        except Exception as e:
            print(f"Amplitude-amplification demo could not be built: {e}")

    if args.noise_model_mode == "simple":
        if ideal_metrics is None:
            state = Statevector.from_instruction(qc)
            ideal_metrics = analyze_statevector_result(profile, np.asarray(state))
        if not benchmarks:
            benchmarks = benchmark_circuit_set(profile, qc, approximation_degree=args.approximation_degree)
        summaries = build_noise_summaries(
            profile, ideal_metrics, benchmarks,
            angle_noise_std=args.angle_noise_std,
            p_two_qubit=args.two_qubit_error,
            readout_p=args.readout_error,
            seed=args.seed,
        )
        print_noise_report(summaries)

    if args.qml_demo:
        print_qml_demo(profile)

    # Reviewer-facing JSON metrics.  This keeps figure and manuscript claims auditable.
    try:
        metrics_payload = {
            "classical_profile": {
                "N": profile.N,
                "M": profile.M,
                "m": profile.m,
                "workspace_mode": profile.workspace_mode,
                "alpha": profile.alpha,
                "p_ancilla_success": profile.p_ancilla_success,
                "expected_repetitions": profile.expected_repetitions,
                "p_logical_unconditional": profile.p_logical_unconditional,
                "p_logical_given_ancilla": profile.p_logical_given_ancilla,
                "aa_repetitions_estimate_only": profile.aa_repetitions,
            },
            "resource_report": resource_report,
            "statevector_metrics": ideal_metrics if ideal_metrics is not None else {},
            "benchmarks": benchmarks,
            "qml_diagnostic": qml_gram_demo(profile.N, samples=4, seed=args.seed),
            "claim_scope": {
                "supports": [
                    "post-selected logical correctness diagnostics",
                    "state-dependent success-probability reporting",
                    "QBA/zero-padding/AQFT small-N benchmark path",
                    "dense exact-QFT_N reference for small N when available",
                    "weak QML feature/kernel discrepancy diagnostic",
                ],
                "does_not_yet_support": [
                    "optimized mixed-radix exact-QFT decomposition",
                    "default amplitude amplification in the main QBA prototype",
                    "hardware-calibrated noise benchmark",
                    "ECG/QCNN application effectiveness",
                    "classifier margin or generalization claims",
                ],
            },
        }
        json_filename = f"qba_metrics_revised_N{args.N}.json"
        write_metrics_json(json_filename, metrics_payload)
        print(f"\n[Metrics] Reviewer-facing JSON metrics saved as '{json_filename}'")
    except Exception as e:
        print(f"\n[Metrics] Could not save JSON metrics: {e}")

    if args.hardware:
        if QiskitRuntimeService is None or Sampler is None:
            raise RuntimeError("qiskit-ibm-runtime is not available; cannot run hardware mode.")
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        tcirc = transpile(qc, backend=backend, optimization_level=3)
        job = Sampler(mode=backend).run([tcirc], shots=args.shots)
        result = job.result()[0]
        counts = result.data.c.get_counts()
        
        hw_metrics = analyze_hardware_counts(profile, counts)
        print("\n=== Hardware post-selection report ===")
        for k, v in hw_metrics.items():
            print(f"{k}: {v}")
            
        # === FIXED: Save hardware histogram automatically ===
        try:
            fig_hw = plot_histogram(hw_metrics["logical_counts"], title=f"Hardware: Bluestein QFT (N={args.N})")
            hw_filename = f'qba_hw_histogram_revised_N{args.N}.png'
            save_figure_tight(fig_hw, Path(hw_filename).with_suffix(''), dpi=600, pad_inches=0.02, save_pdf=True); hw_filename = str(Path(hw_filename).with_suffix('.png'))
            plt.close(fig_hw)
            print(f"\n[Visual] Hardware histogram successfully saved as '{hw_filename}'")
        except Exception as e:
            print(f"\n[Visual] Could not save hardware histogram: {e}")


if __name__ == "__main__":
    main()
