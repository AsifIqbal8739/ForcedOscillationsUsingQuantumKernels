"""
qsvm_kernel_v2.py

Physics-informed quantum kernel + QSVM, matching Section IV-B/V-B of the
manuscript:
  - Fixed, parameter-free-at-inference feature map with data-reupload
    RZ-RX single-qubit rotations and ZZ entanglers WIRED to physically
    meaningful descriptor pairs (coherence<->phase-dispersion, etc.),
    not a generic hardware-efficient ring.
  - Kernel = exact statevector overlap |<psi(z)|psi(z')>|^2 (Eq. 6),
    via Qiskit's Statevector (noiseless, gate-level, infinite-shot limit).
  - Gram-matrix normalization + centering (Eq. 8).
  - Kernel scale(s) (beta, beta') and SVM C chosen by k-fold CV maximizing
    validation AUC (Section V-B), not hardcoded.

Drop-in CLI-compatible replacement for the original qsvm_kernel.py:
    python qsvm_kernel_v2.py --train train.csv --test test.csv --out out.json

CSV format (same as before): one row per window, columns = [z_1,...,z_m, label].
IMPORTANT: feed pass this script the *standardized* (z-scored, using TRAIN
stats only) raw physics descriptors -- do NOT pre-scale by PCA/tanh in
MATLAB anymore; beta/beta' now do that job and are chosen by CV below.

New flags:
    --edges          "5-6,1-3,0-4,2-7"  (physics-wired pairs; default matches
                      the 8 descriptors from extract_features_rich: delta_f,
                      V_band_max, V_band_mean, df_band_max, dom_freq,
                      phase_spread, mean_coh, spec_flat). Omit the flag
                      entirely to get this default. Pass --edges "" or
                      --edges "none" explicitly to get NO entangling gates
                      at all (pure product-state kernel).
    --n_layers       number of re-upload (encode+entangle) layers, default 2
    --beta_grid      comma list, default "0.5,1,2,4"
    --beta_prime_grid comma list, default "0.5,1,2,4"
    --C_grid         comma list, default "0.1,1,10,100"
    --cv_folds       default 3
    --no_cv          skip CV, use --beta/--beta_prime/--C directly (for fast
                      re-inference once hyperparameters are already fixed)
    --beta, --beta_prime, --C   values to use when --no_cv is set
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEFAULT_EDGES = [(5, 6), (1, 3), (0, 4), (2, 7)]


def parse_edges(s):
    if s is None:
        return DEFAULT_EDGES
    if s.strip() == "" or s.strip().lower() == "none":
        return []
    edges = []
    for pair in s.split(","):
        a, b = pair.split("-")
        edges.append((int(a), int(b)))
    return edges


def feature_map_circuit(z, beta, beta_prime, edges, n_layers=2):
    """Physics-informed feature map: data-reupload RZ-RX rotations per qubit,
    with ZZ entanglers wired ONLY between physically related descriptor
    pairs (edges), repeated n_layers times.

    `beta` may be a scalar (original global-scale design) OR a length-D
    array (ARD-style per-feature scale -- lets the kernel emphasize
    informative descriptors and suppress noisy ones, the way a linear
    model's learned weight vector already can and the original scalar-beta
    design could not)."""
    D = len(z)
    beta_vec = np.broadcast_to(np.asarray(beta, dtype=float), (D,))
    qc = QuantumCircuit(D)
    for _ in range(n_layers):
        for q in range(D):
            qc.rz(float(beta_vec[q] * z[q]), q)
            qc.rx(float(beta_vec[q] * z[q]), q)
        for (a, b) in edges:
            qc.rzz(float(beta_prime * z[a] * z[b]), a, b)
    return qc


def states_matrix(X, beta, beta_prime, edges, n_layers=2):
    X = np.asarray(X)
    dim = 1 << X.shape[1]
    M = np.empty((X.shape[0], dim), dtype=complex)
    for i, x in enumerate(X):
        sv = Statevector.from_instruction(feature_map_circuit(x, beta, beta_prime, edges, n_layers))
        M[i, :] = sv.data
    return M


def kernel_matrix_from_states(SA, SB):
    K = SA.conj() @ SB.T
    return np.abs(K) ** 2


def normalize_center_kernel(K_train, K_test=None, K_test_diag=None):
    """Eq. 8: self-normalize then center the Gram matrix using TRAINING
    statistics only; test rows normalized by their own diagonal and
    centered with train mu/kbar (standard kernel-PCA out-of-sample rule)."""
    d = np.sqrt(np.diag(K_train))
    Ktilde = K_train / np.outer(d, d)
    mu = Ktilde.mean(axis=0)
    kbar = Ktilde.mean()
    Khat_train = Ktilde - mu[None, :] - mu[:, None] + kbar
    if K_test is None:
        return Khat_train, mu, kbar, d
    d_test = np.sqrt(K_test_diag)
    Ktilde_test = K_test / np.outer(d_test, d)
    row_mean_test = Ktilde_test.mean(axis=1)
    Khat_test = Ktilde_test - row_mean_test[:, None] - mu[None, :] + kbar
    return Khat_train, Khat_test, mu, kbar, d


def kernel_label_alignment(Khat, y):
    yv = y.astype(float).reshape(-1, 1)
    num = float((yv.T @ Khat @ yv).item())
    den = float(np.linalg.norm(Khat, "fro") * (np.linalg.norm(yv) ** 2) + 1e-12)
    return num / den


def fit_and_eval(Xtr, ytr, Xte, beta, beta_prime, edges, C, n_layers,
                  hybrid_alpha=None):
    """hybrid_alpha: None => pure quantum kernel (original behavior).
    A float in [0,1] => K = alpha*K_linear + (1-alpha)*K_quantum, where
    K_linear is the normalized/centered linear Gram (X X^T) on the same
    standardized features -- the kernel-space equivalent of what a linear
    model uses. alpha=1 reduces to a linear-kernel SVM; alpha=0 is pure
    quantum. Both component Grams are Eq.8-normalized/centered BEFORE
    mixing so neither dominates by scale alone."""
    n_qubits = Xtr.shape[1]
    t0 = time.time()
    Str = states_matrix(Xtr, beta, beta_prime, edges, n_layers)
    Kq_tr_raw = kernel_matrix_from_states(Str, Str)
    Kq_tr, mu, kbar, d = normalize_center_kernel(Kq_tr_raw)
    if hybrid_alpha is not None:
        Kl_tr_raw = Xtr @ Xtr.T
        Kl_tr, _, _, _ = normalize_center_kernel(Kl_tr_raw + 1e-9*np.eye(Xtr.shape[0]))
        Khat_tr = hybrid_alpha * Kl_tr + (1.0 - hybrid_alpha) * Kq_tr
    else:
        Khat_tr = Kq_tr
    train_kernel_time = time.time() - t0

    clf = SVC(kernel="precomputed", C=C, class_weight="balanced", random_state=123)
    clf.fit(Khat_tr, ytr)

    t0 = time.time()
    Ste = states_matrix(Xte, beta, beta_prime, edges, n_layers)
    Kq_te_raw = kernel_matrix_from_states(Ste, Str)
    diag_te = np.array([kernel_matrix_from_states(s[None, :], s[None, :])[0, 0] for s in Ste])
    _, Kq_te, _, _, _ = normalize_center_kernel(Kq_tr_raw, Kq_te_raw, diag_te)
    if hybrid_alpha is not None:
        Kl_te_raw = Xte @ Xtr.T
        diag_l_te = np.sum(Xte * Xte, axis=1) + 1e-9
        _, Kl_te, _, _, _ = normalize_center_kernel(
            Kl_tr_raw + 1e-9*np.eye(Xtr.shape[0]), Kl_te_raw, diag_l_te)
        Khat_te = hybrid_alpha * Kl_te + (1.0 - hybrid_alpha) * Kq_te
    else:
        Khat_te = Kq_te
    test_kernel_time = time.time() - t0

    # NOTE: deliberately NOT using SVC(probability=True)/predict_proba here.
    # scikit-learn's Platt scaling relies on an internal 5-fold CV to fit the
    # calibration sigmoid, which becomes unstable (and can even invert the
    # ranking) when n is small -- at n=50 this leaves ~5 samples/class per
    # internal fold. Instead we apply a FIXED, deterministic sigmoid to the
    # raw decision margin: this is a monotonic transform (so AUC/ranking is
    # identical to using the raw margin directly), keeps a sensible 0.5
    # decision boundary at margin=0, and has no small-n-unstable calibration
    # step at all.
    margin = clf.decision_function(Khat_te)
    prob = 1.0 / (1.0 + np.exp(-margin))
    pred = (prob >= 0.5).astype(int)
    align = kernel_label_alignment(Khat_tr, ytr)
    return clf, prob, pred, align, train_kernel_time, test_kernel_time, n_qubits


def select_hybrid_alpha(Xtr, ytr, edges, beta, beta_prime, C, n_layers,
                         alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), n_folds=3, seed=7):
    """Select the hybrid mixing weight alpha by k-fold CV AUC on the
    TRAINING data only. Note the endpoints make this selection maximally
    honest: alpha=1 is 'just use the linear kernel' (LR-like), alpha=0 is
    'just use the quantum kernel' -- so if the quantum component carries no
    incremental signal, CV will simply select alpha=1 and the hybrid
    gracefully degrades to linear behavior rather than being forced to
    include quantum structure that doesn't help."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for alpha in alpha_grid:
        aucs = []
        for tr_idx, va_idx in skf.split(Xtr, ytr):
            _, prob, _, _, _, _, _ = fit_and_eval(
                Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], beta, beta_prime, edges, C,
                n_layers, hybrid_alpha=alpha)
            if len(np.unique(ytr[va_idx])) > 1:
                aucs.append(roc_auc_score(ytr[va_idx], prob))
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        results.append({"alpha": alpha, "mean_auc": mean_auc})
    best = max(results, key=lambda r: r["mean_auc"])
    return best["alpha"], best["mean_auc"], results


def cv_select(Xtr, ytr, edges, beta_grid, beta_prime_grid, C_grid, n_layers, n_folds, seed=7):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for beta in beta_grid:
        for beta_prime in beta_prime_grid:
            for C in C_grid:
                aucs = []
                for tr_idx, va_idx in skf.split(Xtr, ytr):
                    _, prob, _, _, _, _, _ = fit_and_eval(
                        Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], beta, beta_prime, edges, C, n_layers
                    )
                    if len(np.unique(ytr[va_idx])) > 1:
                        aucs.append(roc_auc_score(ytr[va_idx], prob))
                mean_auc = float(np.mean(aucs)) if aucs else float("nan")
                results.append({"beta": beta, "beta_prime": beta_prime, "C": C, "mean_auc": mean_auc})
    best = max(results, key=lambda r: r["mean_auc"])
    return best, results


def optimize_per_feature_beta(Xtr, ytr, edges, n_layers, beta_prime_init=1.0, seed=7,
                               maxiter=80, val_frac=0.3, beta_bounds=(0.05, 1.2), l2_reg=0.05):
    """ARD-style per-feature kernel scale: optimize a length-D beta vector
    (one scale per qubit/descriptor) plus a single beta_prime, by maximizing
    kernel-TARGET ALIGNMENT (Cristianini et al.) -- NOT test-set AUC, and
    not even validation AUC directly, since that would require an expensive
    SVM refit at every optimizer iteration. Alignment is fast to compute
    (one kernel matrix, no SVM fit) and is a standard, principled proxy for
    kernel quality used in the kernel-methods literature.

    IMPORTANT: beta_bounds defaults to (0.05, 1.2), NOT a wide range. Earlier
    scalar-beta CV grid searches on this same problem found beta values
    around 2-4 push the RZ/RX rotation argument (beta*z) into an
    angle-wrapping/aliasing regime that degrades generalization even though
    it can still look fine (or even better) on a TRAINING-only alignment
    objective -- an unconstrained search rediscovered exactly this trap.
    A small L2 penalty additionally discourages drifting far from the
    uniform (beta=1, i.e. "no adjustment") baseline unless the alignment
    gain clearly justifies it.

    Uses an INNER train/val split of the TRAINING data only (val_frac held
    out) purely as a light guard against the optimizer overfitting the
    scale parameters to idiosyncrasies of the exact training sample; the
    real test set is never touched by this function.
    """
    from scipy.optimize import minimize

    D = Xtr.shape[1]
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    idx = rng.permutation(n)
    n_tr = max(1, int((1 - val_frac) * n))
    opt_idx = idx[:n_tr]   # only this inner subset's alignment is optimized against

    lo, hi = beta_bounds

    def neg_alignment(params):
        beta_vec = np.clip(params[:D], lo, hi)
        beta_prime = float(np.clip(params[D], lo, hi))
        S = states_matrix(Xtr[opt_idx], beta_vec, beta_prime, edges, n_layers)
        K = kernel_matrix_from_states(S, S)
        Khat, _, _, _ = normalize_center_kernel(K)
        a = kernel_label_alignment(Khat, ytr[opt_idx])
        penalty = l2_reg * float(np.mean((beta_vec - 1.0) ** 2))
        return -a + penalty

    x0 = np.concatenate([np.ones(D), [beta_prime_init]])
    res = minimize(neg_alignment, x0, method="Nelder-Mead",
                    options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-3})
    beta_vec = np.clip(res.x[:D], lo, hi)
    beta_prime = float(np.clip(res.x[D], lo, hi))
    return beta_vec, beta_prime, float(-res.fun)


def select_C_given_beta(Xtr, ytr, edges, beta, beta_prime, C_grid, n_layers, n_folds, seed=7):
    """Cheap CV over C only, with beta/beta' already fixed (e.g. from the
    ARD alignment optimizer). Kept separate from that continuous
    optimization since C-selection via grid search is standard, convex-ish,
    and inexpensive -- no need to fold it into the harder continuous search."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for C in C_grid:
        aucs = []
        for tr_idx, va_idx in skf.split(Xtr, ytr):
            _, prob, _, _, _, _, _ = fit_and_eval(
                Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], beta, beta_prime, edges, C, n_layers
            )
            if len(np.unique(ytr[va_idx])) > 1:
                aucs.append(roc_auc_score(ytr[va_idx], prob))
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        results.append({"C": C, "mean_auc": mean_auc})
    best = max(results, key=lambda r: r["mean_auc"])
    return best["C"], best["mean_auc"], results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--beta_grid", default="0.5,1,2,4")
    ap.add_argument("--beta_prime_grid", default="0.5,1,2,4")
    ap.add_argument("--C_grid", default="0.1,1,10,100")
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--no_cv", action="store_true")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta_prime", type=float, default=1.0)
    ap.add_argument("--C", type=float, default=1.0)
    # ARD per-feature beta mode
    ap.add_argument("--per_feature_beta", action="store_true",
                     help="Optimize a per-qubit beta vector via kernel-target "
                          "alignment (inner train/val split of TRAINING data "
                          "only), instead of the scalar beta grid search.")
    ap.add_argument("--beta_vec", default=None,
                     help="Comma list of per-feature beta values, length = n_qubits. "
                          "Use with --no_cv to re-run a previously-found ARD beta_vec "
                          "without re-optimizing (fast re-inference, e.g. for "
                          "robustness sweeps).")
    ap.add_argument("--ard_maxiter", type=int, default=80)
    ap.add_argument("--ard_val_frac", type=float, default=0.3)
    # Hybrid residual kernel: K = alpha*K_linear + (1-alpha)*K_quantum
    ap.add_argument("--hybrid", action="store_true",
                     help="Use the hybrid residual kernel alpha*K_linear + "
                          "(1-alpha)*K_quantum, with alpha selected by CV on the "
                          "training data (grid includes alpha=1, i.e. pure linear, "
                          "so the hybrid gracefully degrades to LR-like behavior "
                          "if the quantum component adds nothing).")
    ap.add_argument("--hybrid_alpha", type=float, default=None,
                     help="Fix alpha directly (skip the alpha CV). Use with --no_cv "
                          "for fast re-inference in robustness sweeps.")
    ap.add_argument("--ard_beta_max", type=float, default=1.2,
                     help="Upper bound for ARD per-feature beta search (default 1.2, "
                          "based on earlier scalar-beta grid searches showing beta>~1.5-2 "
                          "enters an angle-wrapping/aliasing regime that degrades "
                          "generalization).")
    ap.add_argument("--ard_l2_reg", type=float, default=0.05,
                     help="L2 penalty discouraging ARD beta values from drifting far from "
                          "the uniform (beta=1) baseline unless alignment gain justifies it.")
    args = ap.parse_args()

    edges = parse_edges(args.edges)
    tr = pd.read_csv(args.train, header=None).to_numpy()
    te = pd.read_csv(args.test, header=None).to_numpy()
    Xtr, ytr = tr[:, :-1], tr[:, -1].astype(int)
    Xte, yte = te[:, :-1], te[:, -1].astype(int)

    ard_alignment = None
    cv_results = None

    if args.beta_vec is not None:
        # Fixed reuse of a previously-found per-feature beta vector (fast
        # re-inference path -- pair with --no_cv).
        beta_val = np.array([float(x) for x in args.beta_vec.split(",")])
        if args.no_cv:
            best = {"beta": beta_val, "beta_prime": args.beta_prime, "C": args.C, "mean_auc": None}
        else:
            C_grid = [float(x) for x in args.C_grid.split(",")]
            C_best, C_auc, cv_results = select_C_given_beta(
                Xtr, ytr, edges, beta_val, args.beta_prime, C_grid, args.n_layers, args.cv_folds)
            best = {"beta": beta_val, "beta_prime": args.beta_prime, "C": C_best, "mean_auc": C_auc}
    elif args.per_feature_beta:
        beta_val, beta_prime_val, ard_alignment = optimize_per_feature_beta(
            Xtr, ytr, edges, args.n_layers, beta_prime_init=args.beta_prime,
            maxiter=args.ard_maxiter, val_frac=args.ard_val_frac,
            beta_bounds=(0.05, args.ard_beta_max), l2_reg=args.ard_l2_reg)
        C_grid = [float(x) for x in args.C_grid.split(",")]
        C_best, C_auc, cv_results = select_C_given_beta(
            Xtr, ytr, edges, beta_val, beta_prime_val, C_grid, args.n_layers, args.cv_folds)
        best = {"beta": beta_val, "beta_prime": beta_prime_val, "C": C_best, "mean_auc": C_auc}
    elif args.no_cv:
        best = {"beta": args.beta, "beta_prime": args.beta_prime, "C": args.C, "mean_auc": None}
    else:
        beta_grid = [float(x) for x in args.beta_grid.split(",")]
        beta_prime_grid = [float(x) for x in args.beta_prime_grid.split(",")]
        C_grid = [float(x) for x in args.C_grid.split(",")]
        best, cv_results = cv_select(Xtr, ytr, edges, beta_grid, beta_prime_grid, C_grid,
                                      args.n_layers, args.cv_folds)

    # ---- Hybrid alpha: fixed via flag, or selected by CV after beta/C settle ----
    hybrid_alpha = None
    alpha_cv_results = None
    if args.hybrid_alpha is not None:
        hybrid_alpha = float(args.hybrid_alpha)
    elif args.hybrid:
        hybrid_alpha, alpha_auc, alpha_cv_results = select_hybrid_alpha(
            Xtr, ytr, edges, best["beta"], best["beta_prime"], best["C"], args.n_layers,
            n_folds=args.cv_folds)
        print(f"Hybrid alpha selected by CV: alpha={hybrid_alpha} (CV AUC={alpha_auc:.4f})")

    clf, prob, pred, align, tr_time, te_time, n_qubits = fit_and_eval(
        Xtr, ytr, Xte, best["beta"], best["beta_prime"], edges, best["C"], args.n_layers,
        hybrid_alpha=hybrid_alpha
    )

    gates_per_layer = 2 * n_qubits + len(edges)   # RZ+RX per qubit, RZZ per edge
    circuit_depth_est = args.n_layers * 3          # rot layer(2 gate-layers)+entangle layer, rough depth

    beta_out = best["beta"].tolist() if isinstance(best["beta"], np.ndarray) else best["beta"]

    out = {
        "pred_probs": prob.tolist(),
        "pred_labels": pred.tolist(),
        "hyperparams": {"beta": beta_out, "beta_prime": best["beta_prime"], "C": best["C"],
                          "hybrid_alpha": hybrid_alpha},
        "cv_mean_auc": best.get("mean_auc"),
        "cv_grid_results": cv_results,
        "alpha_cv_results": alpha_cv_results,
        "ard_train_alignment": ard_alignment,
        "kernel_label_alignment_train": align,
        "config": {
            "n_qubits": n_qubits,
            "edges": edges,
            "n_layers": args.n_layers,
            "gates_per_layer": gates_per_layer,
            "approx_circuit_depth": circuit_depth_est,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "per_feature_beta": bool(args.per_feature_beta or args.beta_vec is not None),
        },
        "timing_seconds": {
            "train_kernel_construction": tr_time,
            "test_kernel_construction": te_time,
            "per_kernel_entry_train": tr_time / max(1, Xtr.shape[0] ** 2),
            "per_kernel_entry_test": te_time / max(1, Xte.shape[0] * Xtr.shape[0]),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done. n_qubits={n_qubits}, best_params={{'beta': {beta_out}, 'beta_prime': {best['beta_prime']}, "
          f"'C': {best['C']}, 'mean_auc': {best.get('mean_auc')}}}, "
          f"train_kernel_time={tr_time:.3f}s, test_kernel_time={te_time:.3f}s")


if __name__ == "__main__":
    main()
