#!/usr/bin/env python3
"""Reproduce run_isone_final_matched.py exactly and keep per-window predictions.

Optional --variant applies a modified attack/processing at TEST time only
(training is always the published protocol), for the Reviewer-4 robustness controls.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import run_isone_semisynthetic_validation as base
import run_isone_nonstationary_experiments as ns
import run_isone_projected_quantum as pq
import run_isone_final_matched as fm
import qsvm_kernel_v2 as qk
import attack_variants as av

def fold_preds(events, test, extra_models=()):
    X, y, g = fm.stack(events); Xt, yt, pairs = test[0]["structured"], test[1], test[2]
    pred, sel = {}, {}
    for model in ["LR", "RBF"]:
        best, _ = fm.select_classical(X, y, g, model); pred[model] = fm.fit_classical(X, y, Xt, model, best); sel[model] = best
    best, _ = pq.robust_select(X, y, g, "fidelity", qk)
    sc = StandardScaler().fit(X); Z = sc.transform(X); Zt = sc.transform(Xt)
    A, B = pq.kernels(Z, Zt, "fidelity", best, qk); pred["QSVM"] = pq.score_kernel(A, y, B, best["C"]); sel["QSVM"] = best
    for name, fn in extra_models:
        pred[name], sel[name] = fn(X, y, g, Xt)
    return yt, pairs, pred, sel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="400,401,402,403,404"); ap.add_argument("--variant", default="published")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--extra", default="", help="comma list of extra classical models (see extra_models.py)")
    a = ap.parse_args()
    paths = [Path(f"ISO-NE_case{i}.csv") for i in range(1, 6)]
    common = sorted(set.intersection(*[set(base.usable_pmus(p)) for p in paths]))
    raw = [base.read_case(p, keep_indices=common) for p in paths]
    extras = []
    if a.extra:
        import extra_models as em
        extras = [(n, em.grouped(n)) for n in a.extra.split(",")]
    out = {"variant": a.variant, "runs": []}
    for seed in map(int, a.seeds.split(",")):
        ev = [pq.augment_event(r, seed + 10 * i, 3, False) for i, r in enumerate(raw)]
        run = {"seed": seed, "folds": []}
        for h in range(5):
            if a.ckpt and Path(a.ckpt).exists() and any(json.loads(l)["seed"] == seed and json.loads(l)["heldout_case"] == h + 1 for l in open(a.ckpt)):
                continue
            if a.variant == "published":
                te = pq.augment_event(raw[h], seed + 1000 + h, 1, True)
            else:
                te = av.make_test_event(raw[h], seed + 1000 + h, a.variant)
            yt, pairs, pred, sel = fold_preds([ev[i] for i in range(5) if i != h], te, extras)
            run["folds"].append({"heldout_case": h + 1, "y": yt.tolist(), "pairs": pairs.tolist(),
                                 "pred": {k: np.asarray(v).tolist() for k, v in pred.items()},
                                 "auc": {k: float(roc_auc_score(yt, v)) for k, v in pred.items()},
                                 "selected": {k: {kk: (vv if not isinstance(vv, np.generic) else vv.item())
                                                  for kk, vv in s.items() if kk in ("C", "gamma", "beta", "bp", "degree", "hidden", "alpha", "lr", "depth", "D", "omega")}
                                              for k, s in sel.items()}})
            print(seed, h + 1, {k: round(v, 4) for k, v in run["folds"][-1]["auc"].items()}, flush=True)
            if a.ckpt:
                with open(a.ckpt, "a") as fh: fh.write(json.dumps({"seed": seed, **run["folds"][-1]}) + "\n")
        out["runs"].append(run)
    json.dump(out, open(a.out, "w"))

if __name__ == "__main__":
    main()
