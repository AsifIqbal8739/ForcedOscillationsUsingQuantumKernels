#!/usr/bin/env python3
"""Reviewer-4 comment 5: train exactly as run_isone_final_matched.py, then score the held-out
event under each test-time attack/processing variant (attack_variants.py)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base
import run_isone_projected_quantum as pq
import run_isone_final_matched as fm
import qsvm_kernel_v2 as qk
import attack_variants as av

seeds = [int(s) for s in sys.argv[1].split(",")]; OUT = sys.argv[2]
variants = ["published"] + av.VARIANTS
paths = [Path(f"ISO-NE_case{i}.csv") for i in range(1, 6)]
common = sorted(set.intersection(*[set(base.usable_pmus(p)) for p in paths]))
raw = [base.read_case(p, keep_indices=common) for p in paths]
res = []
for seed in seeds:
    ev = [pq.augment_event(r, seed + 10 * i, 3, False) for i, r in enumerate(raw)]
    for h in range(5):
        X, y, g = fm.stack([ev[i] for i in range(5) if i != h])
        sel = {m: fm.select_classical(X, y, g, m)[0] for m in ["LR", "RBF"]}
        qb, _ = pq.robust_select(X, y, g, "fidelity", qk)
        sc = StandardScaler().fit(X); Z = sc.transform(X)
        lr = LogisticRegression(C=sel["LR"]["C"], max_iter=3000, class_weight="balanced").fit(Z, y)
        rb = SVC(C=sel["RBF"]["C"], gamma=sel["RBF"]["gamma"], kernel="rbf", class_weight="balanced").fit(Z, y)
        for v in variants:
            te = av.make_test_event(raw[h], seed + 1000 + h, v)
            Zt = sc.transform(te[0]["structured"]); yt = te[1]
            A, B = pq.kernels(Z, Zt, "fidelity", qb, qk)
            pred = {"LR": lr.predict_proba(Zt)[:, 1], "RBF": base.sigmoid(rb.decision_function(Zt)),
                    "QSVM": pq.score_kernel(A, y, B, qb["C"])}
            res.append({"seed": seed, "heldout_case": h + 1, "variant": v, "y": yt.tolist(), "pairs": te[2].tolist(),
                        "pred": {k: p.tolist() for k, p in pred.items()},
                        "auc": {k: float(roc_auc_score(yt, p)) for k, p in pred.items()}})
        print(seed, h + 1, {v: round(r["auc"]["QSVM"], 3) for r in res[-len(variants):] for v in [r["variant"]]}, flush=True)
json.dump(res, open(OUT, "w"))
