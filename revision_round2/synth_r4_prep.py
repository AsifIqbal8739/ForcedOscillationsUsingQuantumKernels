#!/usr/bin/env python3
"""(prep only) Reviewer-4 experiments on the FROZEN 20-seed shifted-footprint experiment
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

import os
ROOT = Path(os.environ.get('TCE_SYNTH_DATA', '../synthetic/data/stageB_prime_confirmatory'))
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

