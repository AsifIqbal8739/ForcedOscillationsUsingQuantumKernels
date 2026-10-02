#!/usr/bin/env python3
"""Reviewer-4 experiments on the FROZEN 20-seed shifted-footprint experiment
(stageB_prime_confirmatory, phase representation; preprocessing identical to
run_stageB_prime.py).

  part 'baselines': extra classical models on the identical 400-sample, 8-D inputs.
  part 'topology' : QSVM with no / headline (DEFAULT_EDGES) / ring entanglement on the
                    identical inputs and splits; each topology re-selects beta, beta', C
                    with the paper's grid and 3-fold CV.
  part 'full'     : LR, RBF-SVM and QSVM trained on all 2000 training windows.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
import qsvm_kernel_v2 as qk
import extra_models as em

ROOT = Path(sys.argv[1]); part = sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]; OUT = sys.argv[4]
def load(p):
    a = pd.read_csv(p, header=None).to_numpy(); return a[:, :-1], a[:, -1].astype(int)
def sig(x): return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

def prep(seed, cap=400):
    X, y = load(ROOT / f"data_seed{seed}/train_phase.csv"); Xt, yt = load(ROOT / f"data_seed{seed}/shifted_footprint_phase.csv")
    if cap and len(y) > cap:
        rng = np.random.default_rng(seed)
        ix = np.concatenate([rng.choice(np.flatnonzero(y == c), cap // 2, replace=False) for c in [0, 1]]); rng.shuffle(ix)
        X, y = X[ix], y[ix]
    sc = StandardScaler().fit(X); Z = sc.transform(X); Zt = sc.transform(Xt)
    p = PCA(n_components=8, random_state=seed).fit(Z); Z = p.transform(Z); Zt = p.transform(Zt)
    s2 = StandardScaler().fit(Z); return s2.transform(Z), y, s2.transform(Zt), yt

def classical(Z, y, Zt, yt, seed):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    lr = GridSearchCV(LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [.1, 1, 10, 100]}, cv=cv, scoring="roc_auc").fit(Z, y)
    rb = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [.1, 1, 10, 100], "gamma": ["scale", .1, 1]}, cv=cv, scoring="roc_auc").fit(Z, y)
    return {"LR": roc_auc_score(yt, lr.predict_proba(Zt)[:, 1]), "RBF": roc_auc_score(yt, sig(rb.decision_function(Zt)))}

def qsvm(Z, y, Zt, yt, seed, edges):
    best, _ = qk.cv_select(Z, y, edges, [.5, 1, 2], [.5, 1, 2], [.1, 1, 10], 2, 3, seed)
    _, p, *_ = qk.fit_and_eval(Z, y, Zt, best["beta"], best["beta_prime"], edges, best["C"], 2, None)
    return roc_auc_score(yt, p), {k: best[k] for k in ("beta", "beta_prime", "C")}

TOPO = {"none": [], "headline": qk.DEFAULT_EDGES, "ring": [(i, (i + 1) % 8) for i in range(8)]}
res = []
for seed in seeds:
    t0 = time.time(); r = {"seed": seed}
    if part in ("baselines", "topology"):
        Z, y, Zt, yt = prep(seed)
    if part == "baselines":
        r["auc"] = classical(Z, y, Zt, yt, seed); r["sel"] = {}
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        for name, (est, params, how) in em.specs(seed).items():
            g = GridSearchCV(clone(est), params, cv=cv, scoring="roc_auc").fit(Z, y)
            r["auc"][name] = roc_auc_score(yt, em.score(g.best_estimator_, Zt, how))
            r["sel"][name] = {k: str(v) for k, v in g.best_params_.items()}
    elif part == "topology":
        r["auc"], r["sel"] = {}, {}
        for name, edges in TOPO.items():
            r["auc"][name], r["sel"][name] = qsvm(Z, y, Zt, yt, seed, edges)
    elif part == "full":
        Z, y, Zt, yt = prep(seed, cap=None); r["n_train"] = int(len(y))
        r["auc"] = classical(Z, y, Zt, yt, seed)
        r["auc"]["QSVM"], r["sel"] = qsvm(Z, y, Zt, yt, seed, qk.DEFAULT_EDGES)
    r["sec"] = time.time() - t0; res.append(r)
    print(seed, {k: round(v, 4) for k, v in r["auc"].items()}, f"{r['sec']:.0f}s", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
