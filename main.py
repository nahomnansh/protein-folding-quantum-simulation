"""
Protein Folding with VQE (2D HP Lattice Model)
----------------------------------------------
Problem: a protein is a chain of amino acids that folds into a 3D shape which
determines its function. Predicting the lowest-energy fold is a famously hard
optimization problem (it is NP-hard even in simplified models).

Model used here (the classic "HP model"):
  * Each amino acid is Hydrophobic (H) or Polar (P).
  * The chain lives on a 2D square lattice, one residue per lattice site.
  * Energy = -1 for every pair of H residues that touch on the lattice but are
    not neighbours along the chain (H's like to bury themselves together).
  * Two residues on the same site is illegal -> large penalty.

Quantum encoding:
  * Each bond between residues points in one of 4 directions -> 2 qubits/bond.
  * The first bond is fixed (removes rotational symmetry), so an N-residue
    chain needs 2 * (N - 2) qubits.
  * The energy is diagonal in the computational basis, so we write it as a
    Hamiltonian made of Pauli-Z strings and minimise <H> with VQE.

Everything is compared against a brute-force classical search.

Install:  pip install qiskit numpy scipy matplotlib
Run:      python protein_folding_vqe.py
          (you will be prompted for each setting; press Enter for defaults)
"""

import time
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, up, left, down
OVERLAP_PENALTY = 10.0


# ---------------------------------------------------------------------------
# 1. Physics: turn a bitstring into a fold and score it
# ---------------------------------------------------------------------------
def decode_fold(x: int, n_res: int) -> list:
    """
    Convert an integer (the measured bitstring) into lattice coordinates.
    Bond 0 is fixed to 'right'; each later bond uses 2 bits of x.
    """
    dirs = [0] + [(x >> (2 * k)) & 3 for k in range(n_res - 2)]
    pos = [(0, 0)]
    for d in dirs:
        dx, dy = DIRECTIONS[d]
        pos.append((pos[-1][0] + dx, pos[-1][1] + dy))
    return pos


def fold_energy(positions: list, seq: str) -> tuple:
    """Return (energy, n_contacts, n_overlaps) for a fold."""
    n = len(seq)
    overlaps = sum(
        positions[i] == positions[j] for i in range(n) for j in range(i + 1, n)
    )
    contacts = 0
    for i in range(n):
        for j in range(i + 3, n):  # skip chain neighbours
            if seq[i] == "H" and seq[j] == "H":
                dist = abs(positions[i][0] - positions[j][0]) + abs(
                    positions[i][1] - positions[j][1]
                )
                if dist == 1:
                    contacts += 1
    return -contacts + OVERLAP_PENALTY * overlaps, contacts, overlaps


def energy_table(seq: str) -> np.ndarray:
    """Energy of every possible bitstring (the diagonal of the Hamiltonian)."""
    n_qubits = 2 * (len(seq) - 2)
    return np.array(
        [fold_energy(decode_fold(x, len(seq)), seq)[0] for x in range(2 ** n_qubits)]
    )


# ---------------------------------------------------------------------------
# 2. Hamiltonian: diagonal energies -> sum of Pauli-Z strings
# ---------------------------------------------------------------------------
def walsh_hadamard(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform (unnormalised)."""
    a = a.astype(float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            x = a[i : i + h].copy()
            y = a[i + h : i + 2 * h].copy()
            a[i : i + h] = x + y
            a[i + h : i + 2 * h] = x - y
        h *= 2
    return a


def build_hamiltonian(diag: np.ndarray, n_qubits: int) -> SparsePauliOp:
    """
    Any diagonal operator = sum over masks m of c_m * Z^m, where
    c_m = (1/2^n) * sum_x diag[x] * (-1)^popcount(x & m).
    """
    coeffs = walsh_hadamard(diag) / len(diag)
    labels, weights = [], []
    for mask, c in enumerate(coeffs):
        if abs(c) < 1e-9:
            continue
        # qubit i is the i-th bit of the mask; Qiskit labels put qubit 0 on the right
        label = "".join("Z" if (mask >> i) & 1 else "I" for i in reversed(range(n_qubits)))
        labels.append(label)
        weights.append(c)
    return SparsePauliOp(labels, weights)


# ---------------------------------------------------------------------------
# 3. VQE (hand-built so it works across Qiskit versions)
# ---------------------------------------------------------------------------
def build_ansatz(n_qubits: int, reps: int) -> tuple:
    """RY rotations + linear CNOT entanglement, repeated `reps` times."""
    n_params = n_qubits * (reps + 1)
    thetas = np.zeros(n_params)  # placeholder, real values bound later

    def make(params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        idx = 0
        for _ in range(reps):
            for q in range(n_qubits):
                qc.ry(params[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        return qc

    return make, n_params


def run_vqe(diag: np.ndarray, n_qubits: int, reps: int, restarts: int, seed: int):
    """Minimise <psi(theta)|H|psi(theta)> with several random restarts."""
    make_circuit, n_params = build_ansatz(n_qubits, reps)
    rng = np.random.default_rng(seed)

    best = {"energy": np.inf, "params": None, "history": []}

    for r in range(restarts):
        history = []

        def cost(params):
            probs = Statevector(make_circuit(params)).probabilities()
            e = float(probs @ diag)  # <H> for a diagonal H
            history.append(e)
            return e

        x0 = rng.uniform(-np.pi, np.pi, n_params)
        res = minimize(cost, x0, method="COBYLA", options={"maxiter": 600})
        print(f"  restart {r + 1}/{restarts}: final <H> = {res.fun:+.3f}")
        if res.fun < best["energy"]:
            best = {"energy": res.fun, "params": res.x, "history": history}

    return best, make_circuit


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------
def draw_fold(ax, positions, seq, title):
    xs, ys = zip(*positions)
    ax.plot(xs, ys, "-", color="gray", lw=2, zorder=1)
    for i, (x, y) in enumerate(positions):
        ax.scatter(x, y, s=500, zorder=2, color="#e4572e" if seq[i] == "H" else "#4c8bf5")
        ax.text(x, y, f"{seq[i]}{i}", ha="center", va="center", color="white", fontsize=9, zorder=3)
    # highlight HH contacts
    for i in range(len(seq)):
        for j in range(i + 3, len(seq)):
            if seq[i] == seq[j] == "H":
                d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                if d == 1:
                    ax.plot(
                        [positions[i][0], positions[j][0]],
                        [positions[i][1], positions[j][1]],
                        "--", color="green", lw=2.5, zorder=1,
                    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def ask(prompt: str, default, cast=str):
    """Prompt the user; Enter (or no input available) returns the default."""
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
        except EOFError:  # no stdin available (some browser IDEs)
            print(f"(no input available, using {default})")
            return default
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"  Please enter a valid {cast.__name__}.")


def main():
    print("=== VQE Protein Folding (press Enter to accept [defaults]) ===")
    args = SimpleNamespace(
        seq=ask("Amino acid sequence of H/P letters", "HHPPHH", str),
        reps=ask("Ansatz depth (reps)", 3, int),
        restarts=ask("Number of VQE restarts", 5, int),
        shots=ask("Samples drawn from final state (shots)", 2000, int),
        seed=ask("Random seed", 7, int),
    )
    print()

    seq = args.seq.upper()
    if set(seq) - {"H", "P"} or len(seq) < 4:
        raise SystemExit("Sequence must be at least 4 letters of H and P.")
    n_res = len(seq)
    n_qubits = 2 * (n_res - 2)
    if n_qubits > 14:
        raise SystemExit("Too many qubits for statevector simulation; use <= 8 residues.")

    print(f"Sequence: {seq}  ({n_res} residues -> {n_qubits} qubits)")

    # Energy landscape + classical baseline (brute force)
    t0 = time.time()
    diag = energy_table(seq)
    best_x = int(np.argmin(diag))
    classical_time = time.time() - t0
    exact_energy = diag[best_x]
    n_valid = int(np.sum(diag <= 0))
    print(f"\n[Classical brute force] {2 ** n_qubits} folds checked, {n_valid} valid")
    print(f"  ground-state energy = {exact_energy:+.0f}  ({classical_time:.3f}s)")

    # Hamiltonian
    H = build_hamiltonian(diag, n_qubits)
    print(f"\n[Hamiltonian] {len(H)} Pauli-Z terms on {n_qubits} qubits")
    if n_qubits <= 10:  # sanity check: Pauli form must match the diagonal
        probe = Statevector.from_int(best_x, 2 ** n_qubits)
        assert abs(probe.expectation_value(H).real - exact_energy) < 1e-6
        print("  sanity check passed: <H> on the exact ground state matches")

    # VQE
    print(f"\n[VQE] ansatz reps={args.reps}, restarts={args.restarts}")
    t0 = time.time()
    best, make_circuit = run_vqe(diag, n_qubits, args.reps, args.restarts, args.seed)
    vqe_time = time.time() - t0

    final_state = Statevector(make_circuit(best["params"]))
    probs = final_state.probabilities()
    p_ground = float(probs[diag == exact_energy].sum())

    # Sample like a real device would, keep the best legal fold we see
    counts = final_state.sample_counts(shots=args.shots)
    sampled = {int(k, 2): v for k, v in counts.items()}
    vqe_x = min(sampled, key=lambda k: diag[k])

    print(f"\n  best <H>                 = {best['energy']:+.3f}")
    print(f"  best sampled fold energy = {diag[vqe_x]:+.0f}")
    print(f"  P(ground state)          = {p_ground:.1%}")
    print(f"  VQE wall time            = {vqe_time:.1f}s")
    found = diag[vqe_x] == exact_energy
    print("  RESULT:", "VQE found the optimal fold!" if found else "VQE found a sub-optimal fold (try more reps/restarts).")

    # Plot
    vqe_pos = decode_fold(vqe_x, n_res)
    exact_pos = decode_fold(best_x, n_res)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    draw_fold(axes[0], vqe_pos, seq, f"VQE fold (E = {diag[vqe_x]:+.0f})")
    draw_fold(axes[1], exact_pos, seq, f"Brute-force optimum (E = {exact_energy:+.0f})")
    axes[2].plot(best["history"], color="#6a4c93")
    axes[2].axhline(exact_energy, color="green", ls="--", label="true ground energy")
    axes[2].set_title("VQE convergence (best restart)")
    axes[2].set_xlabel("function evaluation")
    axes[2].set_ylabel("<H>")
    axes[2].legend()
    fig.suptitle(f"HP protein folding: {seq}   (red = hydrophobic, blue = polar, green dashes = H-H contacts)")
    plt.tight_layout()
    plt.savefig("protein_folding_result.png", dpi=150)
    print("\nSaved plot to protein_folding_result.png")
    if matplotlib.get_backend().lower() != "agg":
        try:
            plt.show()
        except Exception:
            pass  # headless / browser IDE: the PNG was already saved


if __name__ == "__main__":
    main()
