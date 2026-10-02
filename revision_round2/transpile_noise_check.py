#!/usr/bin/env python3
"""Run ON YOUR MAC in the qiskit_matlab environment (the only place Qiskit is installed).

  /opt/anaconda3/envs/qiskit_matlab/bin/pip install qiskit-aer qiskit-ibm-runtime
  cd ~/Library/CloudStorage/OneDrive-Personal/MATLAB/Muneeb/TCE/revision_GPT
  /opt/anaconda3/envs/qiskit_matlab/bin/python3 transpile_noise_check.py

Uses the frozen experiment exactly as run_stageB_prime.py (phase features, seeds 100-119,
same 400-sample subset/PCA, qsvm_kernel_v2.feature_map_circuit, DEFAULT_EDGES, stored
beta/beta'). For each seed it:
  1. transpiles 20 random compute-uncompute kernel circuits to FakeTorino (opt. level 3)
     and records depth, native 2-qubit gate count and calibrated-error fidelity F;
  2. runs 30 random kernel entries (8192 shots) under FakeTorino's Aer noise model and
     fits p0 = F_fit * k_exact + c (checks the depolarizing model of Eq. 19).
Writes transpile_noise_check.json next to this script (~5-10 min on an M4 Max).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
import qsvm_kernel_v2 as qk

HERE = Path(__file__).resolve().parent
ROOT = HERE / "stageB_prime_confirmatory"
BACKEND = FakeTorino(); SIM = AerSimulator.from_backend(BACKEND)
N_T, N_A, SHOTS = 20, 30, 8192

def prep(seed):
    def load(p):
        a = pd.read_csv(p, header=None).to_numpy(); return a[:, :-1], a[:, -1].astype(int)
    X, y = load(ROOT / f"data_seed{seed}/train_phase.csv")
    rng = np.random.default_rng(seed)
    ix = np.concatenate([rng.choice(np.flatnonzero(y == c), 200, replace=False) for c in [0, 1]]); rng.shuffle(ix)
    X = X[ix]; Z = StandardScaler().fit_transform(X)
    Z = PCA(n_components=8, random_state=seed).fit_transform(Z)
    return StandardScaler().fit_transform(Z)

def kc(z, zp, b, bp):
    qc = qk.feature_map_circuit(zp, b, bp, qk.DEFAULT_EDGES, 2)
    qc = qc.compose(qk.feature_map_circuit(z, b, bp, qk.DEFAULT_EDGES, 2).inverse()); qc.measure_all(); return qc

def fidelity(t):
    F = 1.0
    for ci in t.data:
        n = ci.operation.name
        if n in ("barrier", "delay"): continue
        qa = tuple(t.find_bit(x).index for x in ci.qubits)
        try: e = BACKEND.target[n][qa].error
        except Exception: e = None
        if e: F *= 1 - e
    return F

out = {"backend": BACKEND.name, "runs": []}
for seed in range(100, 120):
    hp = json.load(open(ROOT / f"stageB_prime_seed{seed}.json"))["feature_sets"]["phase"]["qsvm_hyperparams"]
    b, bp = hp["beta"], hp["beta_prime"]; Z = prep(seed); rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(len(Z), 2, replace=False)) for _ in range(N_T + N_A)]
    rows = []
    for i, j in pairs[:N_T]:
        t = transpile(kc(Z[i], Z[j], b, bp), backend=BACKEND, optimization_level=3, seed_transpiler=int(i * 1000 + j))
        rows.append((t.depth(), sum(1 for c in t.data if c.operation.num_qubits == 2), fidelity(t)))
    d, n2, F = map(np.array, zip(*rows))
    circ = [kc(Z[i], Z[j], b, bp) for i, j in pairs[N_T:]]
    k = [abs(np.vdot(Statevector(qk.feature_map_circuit(Z[i], b, bp, qk.DEFAULT_EDGES, 2)).data,
                     Statevector(qk.feature_map_circuit(Z[j], b, bp, qk.DEFAULT_EDGES, 2)).data)) ** 2
         for i, j in pairs[N_T:]]
    tc = transpile(circ, backend=BACKEND, optimization_level=3, seed_transpiler=7)
    res = SIM.run(tc, shots=SHOTS, seed_simulator=11).result()
    p0 = [res.get_counts(m).get("0" * 8, 0) / SHOTS for m in range(len(tc))]
    slope, icpt = np.polyfit(k, p0, 1)
    r = {"seed": seed, "depth": float(d.mean()), "twoq": float(n2.mean()), "F_calib": float(F.mean()),
         "F_fit": float(slope), "intercept": float(icpt), "corr": float(np.corrcoef(k, p0)[0, 1]),
         "logical_rzz": 2 * 2 * len(qk.DEFAULT_EDGES)}
    out["runs"].append(r); print(r, flush=True)
agg = {key: float(np.mean([r[key] for r in out["runs"]])) for key in ["depth", "twoq", "F_calib", "F_fit", "corr"]}
out["aggregate"] = agg; print("AGGREGATE", agg)
json.dump(out, open(HERE / "transpile_noise_check.json", "w"), indent=1)
