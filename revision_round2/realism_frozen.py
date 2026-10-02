#!/usr/bin/env python3
"""Shot-noise / device-noise / cost analysis on the FROZEN 20-seed shifted-footprint
experiment (stageB_prime_confirmatory, phase representation).

Step 1 reproduces run_stageB_prime.py exactly (same 400-sample subset, scaler,
PCA(8, random_state=seed), re-standardization, DEFAULT_EDGES, 2 layers, the QSVM
hyperparameters stored in each seed's JSON, and the classical GridSearchCV), and
checks every AUC against the stored stageB_prime_seed*.json values.
Step 2 re-evaluates QSVM with finite-shot and depolarized kernel estimates.
Step 3 measures time / peak memory / solver iterations for LR, RBF-SVM, QSVM.
"""
import json, sys, time, tracemalloc, platform, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(sys.argv[1]); REF = json.load(open(sys.argv[2])); OUT = sys.argv[3]
EDGES = [(5, 6), (1, 3), (0, 4), (2, 7)]; LAYERS = 2; MAXTR = 400
SHOTS = [256, 1024, 4096, 16384]; FS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]; REPEATS = 10

def load(p):
    a = pd.read_csv(p, header=None).to_numpy(); return a[:, :-1], a[:, -1].astype(int)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

# ---- feature map identical to qsvm_kernel_v2.feature_map_circuit (Qiskit conventions)
def states(Z, beta, bp):
    N, q = Z.shape; D = 1 << q; idx = np.arange(D)
    psi = np.zeros((N, D), complex); psi[:, 0] = 1
    for _ in range(LAYERS):
        for i in range(q):
            th = (beta * Z[:, i])[:, None]; z = (1 - 2 * ((idx >> i) & 1))[None, :]
            psi = psi * np.exp(-0.5j * th * z)                                        # RZ
            psi = np.cos(th / 2) * psi - 1j * np.sin(th / 2) * psi[:, idx ^ (1 << i)]  # RX
        for a, b in EDGES:
            th = (bp * Z[:, a] * Z[:, b])[:, None]
            zz = ((1 - 2 * ((idx >> a) & 1)) * (1 - 2 * ((idx >> b) & 1)))[None, :]
            psi = psi * np.exp(-0.5j * th * zz)                                        # RZZ
    return psi
def kern(A, B): return np.abs(A.conj() @ B.T) ** 2

# ---- normalize_center_kernel identical to qsvm_kernel_v2 (test centred by its own row mean)
def norm_center(Ktr, Kte, dtr, dte):
    d = np.sqrt(dtr); Kt = Ktr / np.outer(d, d); mu = Kt.mean(0); kb = Kt.mean()
    Htr = Kt - mu[None, :] - mu[:, None] + kb
    Ke = Kte / np.outer(np.sqrt(dte), d)
    Hte = Ke - Ke.mean(1)[:, None] - mu[None, :] + kb
    return Htr, Hte
def psd(K):
    K = 0.5 * (K + K.T); w, V = np.linalg.eigh(K); return (V * np.clip(w, 0, None)) @ V.T

def qsvm_auc(Htr, Hte, ytr, yte, C):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced", random_state=123).fit(Htr, ytr)
    return roc_auc_score(yte, sigmoid(clf.decision_function(Hte)))

def estimate(Ktr, Kte, q, S, F, rng):
    c = (1 - F) * 2.0 ** -q
    mtr, mte = F * Ktr + c, F * Kte + c; dself = F + c
    if S is None:
        Etr, Ete = mtr.copy(), mte; dtr = np.full(len(Ktr), dself); dte = np.full(len(Kte), dself)
    else:
        Etr = rng.binomial(S, np.clip(mtr, 0, 1)) / S; Etr = np.triu(Etr, 1); Etr = Etr + Etr.T
        Ete = rng.binomial(S, np.clip(mte, 0, 1)) / S
        dtr = np.maximum(rng.binomial(S, dself, len(Ktr)) / S, 1 / S)
        dte = np.maximum(rng.binomial(S, dself, len(Kte)) / S, 1 / S)
    np.fill_diagonal(Etr, dtr)
    Htr, Hte = norm_center(Etr, Ete, dtr, dte)
    return psd(Htr), Hte

def med_t(fn, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return r, float(np.median(ts))
def peak(fn):
    tracemalloc.start(); fn(); p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return p / 2**20

runs = []
for seed in range(100, 120):
    d = ROOT / f"data_seed{seed}"; ref = REF[str(seed)]
    X, y = load(d / "train_phase.csv"); Xt, yt = load(d / "shifted_footprint_phase.csv")
    rng = np.random.default_rng(seed)
    ix = np.concatenate([rng.choice(np.flatnonzero(y == c), MAXTR // 2, replace=False) for c in [0, 1]])
    rng.shuffle(ix); X, y = X[ix], y[ix]
    sc = StandardScaler().fit(X); Z = sc.transform(X); Zt = sc.transform(Xt)
    pca = PCA(n_components=8, random_state=seed).fit(Z); Z = pca.transform(Z); Zt = pca.transform(Zt)
    sc2 = StandardScaler().fit(Z); Z = sc2.transform(Z); Zt = sc2.transform(Zt)

    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    lr = GridSearchCV(LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [.1, 1, 10, 100]},
                      cv=cv, scoring="roc_auc").fit(Z, y)
    rbf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [.1, 1, 10, 100], "gamma": ["scale", .1, 1]},
                       cv=cv, scoring="roc_auc").fit(Z, y)
    a_lr = roc_auc_score(yt, lr.predict_proba(Zt)[:, 1]); a_rbf = roc_auc_score(yt, sigmoid(rbf.decision_function(Zt)))

    hp = ref["hp"]; beta, bp, C = hp["beta"], hp["beta_prime"], hp["C"]; q = Z.shape[1]
    St, Se = states(Z, beta, bp), states(Zt, beta, bp); Ktr, Kte = kern(St, St), kern(Se, St)
    Htr, Hte = norm_center(Ktr, Kte, np.diag(Ktr), np.ones(len(Kte)))
    a_q = qsvm_auc(Htr, Hte, y, yt, C)
    run = {"seed": seed, "hp": hp, "auc_repro": {"LR": a_lr, "RBF": a_rbf, "QSVM": a_q},
           "auc_ref": ref["auc"], "maxdiff_vs_ref": max(abs(a_lr - ref["auc"]["LR"]), abs(a_rbf - ref["auc"]["RBF"]),
                                                       abs(a_q - ref["auc"]["QSVM"]))}
    off = Ktr[np.triu_indices(len(Ktr), 1)]; m = float(np.median(off))
    run["median_offdiag_k"] = m; run["mean_offdiag_k"] = float(off.mean())
    run["rel_shot_err_1024_pct"] = 100 * np.sqrt(m * (1 - m) / 1024) / m

    r = np.random.default_rng(10_000 + seed); grid = {}
    for F in FS:
        grid[F] = {"inf": qsvm_auc(*estimate(Ktr, Kte, q, None, F, r), y, yt, C)}
        for S in SHOTS:
            grid[F][S] = float(np.mean([qsvm_auc(*estimate(Ktr, Kte, q, S, F, r), y, yt, C) for _ in range(REPEATS)]))
    run["auc_grid"] = grid

    # ---- cost: final fits with selected hyperparameters, identical inputs
    Clr = lr.best_params_["C"]; Crb, gr = rbf.best_params_["C"], rbf.best_params_["gamma"]
    f_lr = lambda: LogisticRegression(C=Clr, max_iter=3000, class_weight="balanced").fit(Z, y)
    m_lr, t_lr = med_t(f_lr); _, p_lr = med_t(lambda: m_lr.predict_proba(Zt))
    f_rb = lambda: SVC(kernel="rbf", C=Crb, gamma=gr, class_weight="balanced").fit(Z, y)
    m_rb, t_rb = med_t(f_rb); _, p_rb = med_t(lambda: m_rb.decision_function(Zt))
    def f_q():
        S_ = states(Z, beta, bp); K_ = kern(S_, S_); d_ = np.diag(K_)
        H_, _ = norm_center(K_, K_[:1], d_, d_[:1])
        return S_, K_, SVC(kernel="precomputed", C=C, class_weight="balanced").fit(H_, y)
    (S_, K_, m_q), t_q = med_t(f_q)
    def p_q():
        Se_ = states(Zt, beta, bp); B_ = kern(Se_, S_); _, Hb = norm_center(K_, B_, np.diag(K_), np.ones(len(B_)))
        return m_q.decision_function(Hb)
    _, tp_q = med_t(p_q)
    _, t_kq = med_t(lambda: kern(S_, S_)); _, t_sp = med_t(lambda: states(Z, beta, bp))
    run["cost"] = {
        "LR": {"train_s": t_lr, "infer_s": p_lr, "peak_MiB": peak(f_lr), "iter": int(np.max(m_lr.n_iter_)),
               "hp": {"C": Clr}},
        "RBF": {"train_s": t_rb, "infer_s": p_rb, "peak_MiB": peak(f_rb), "iter": int(np.sum(m_rb.n_iter_)),
                "nSV": int(m_rb.n_support_.sum()), "hp": {"C": Crb, "gamma": gr}},
        "QSVM": {"train_s": t_q, "infer_s": tp_q, "state_prep_s": t_sp, "gram_s": t_kq, "peak_MiB": peak(f_q),
                 "iter": int(np.sum(m_q.n_iter_)), "nSV": int(m_q.n_support_.sum()),
                 "qiskit_train_kernel_s_M4Max": ref["t"]["qsvm_train"], "qiskit_test_kernel_s_M4Max": ref["t"]["qsvm_test"]},
        "n_train": int(len(y)), "n_test": int(len(yt)),
        "hw_train_circuits": int(len(y) * (len(y) - 1) // 2), "hw_test_circuits": int(len(yt) * len(y))}
    runs.append(run)
    print(f"seed {seed}: repro LR {a_lr:.4f} RBF {a_rbf:.4f} QSVM {a_q:.4f} | ref QSVM {ref['auc']['QSVM']:.4f}"
          f" | maxdiff {run['maxdiff_vs_ref']:.2e} | S=1024 {grid[1.0][1024]:.4f}", flush=True)

json.dump({"host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version()},
           "runs": runs}, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT)
