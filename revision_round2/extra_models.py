"""Additional classical baselines requested by Reviewer 4 (comment 2).

All models receive exactly the inputs given to LR / RBF-SVM / QSVM. Two selection
protocols mirror the paper: StratifiedKFold(3) GridSearchCV for the frozen synthetic
experiment, and event-grouped leave-one-event-out selection with the balanced
objective J = 0.5*(mean + worst event AUC) for ISO-NE (run_isone_final_matched.py).
"""
import itertools, numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def _sig(x): return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

class PrecomputedKernelSVC(ClassifierMixin, BaseEstimator):
    """SVC on a classical kernel given in closed form; kind in {'laplacian','cosprod'}."""
    def __init__(self, kind="laplacian", C=1.0, gamma=0.1, omega=1.0):
        self.kind, self.C, self.gamma, self.omega = kind, C, gamma, omega
    def _k(self, A, B):
        if self.kind == "laplacian":   # exp(-gamma * ||a-b||_1)
            return np.exp(-self.gamma * np.abs(A[:, None, :] - B[None, :, :]).sum(-1))
        # periodic product kernel: prod_i (1+cos(omega (a_i-b_i)))/2 = prod_i cos^2(omega (a_i-b_i)/2)
        return np.prod(np.cos(0.5 * self.omega * (A[:, None, :] - B[None, :, :])) ** 2, axis=-1)
    def fit(self, X, y):
        self.X_ = np.asarray(X); self.classes_ = np.unique(y)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(self._k(self.X_, self.X_), y)
        return self
    def decision_function(self, X): return self.svc_.decision_function(self._k(np.asarray(X), self.X_))
    def predict(self, X): return self.classes_[(self.decision_function(X) > 0).astype(int)]

C4 = [0.1, 1, 10, 100]
def specs(seed):
    """name -> (estimator, param grid, score function). Grids follow the paper's C grid."""
    return {
        "Poly-SVM": (SVC(kernel="poly", coef0=1.0, gamma="scale", class_weight="balanced"),
                     {"C": C4, "degree": [2, 3]}, "dec"),
        "Laplacian-SVM": (PrecomputedKernelSVC("laplacian"), {"C": C4, "gamma": [1/32, 1/16, 1/8, 1/4]}, "dec"),
        "MLP": (MLPClassifier(max_iter=3000, random_state=seed),
                {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 32)], "alpha": [1e-4, 1e-2, 1]}, "proba"),
        "GBoost": (HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"),
                   {"learning_rate": [0.05, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300]}, "proba"),
        "RFF-256": (Pipeline([("rff", RBFSampler(n_components=256, random_state=seed)),
                              ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
                    {"rff__gamma": [0.05, 0.1, 0.5, 1.0], "lr__C": C4}, "proba"),
        "CosProd-SVM": (PrecomputedKernelSVC("cosprod"), {"C": C4, "omega": [0.25, 0.5, 1.0, 2.0]}, "dec"),
    }

def score(est, X, how):
    return est.predict_proba(X)[:, 1] if how == "proba" else _sig(est.decision_function(X))

def grid(params):
    keys = list(params); return [dict(zip(keys, v)) for v in itertools.product(*params.values())]

def grouped(name, seed=0, small=True):
    """Event-grouped selection with J (ISO-NE protocol); inputs are standardized on training events."""
    def run(X, y, g, Xt):
        est0, params, how = (specs_small(seed) if small else specs(seed))[name]; best = None
        for p in grid(params):
            aa = []
            for tr, va in LeaveOneGroupOut().split(X, y, g):
                sc = StandardScaler().fit(X[tr]); m = clone(est0).set_params(**p).fit(sc.transform(X[tr]), y[tr])
                aa.append(roc_auc_score(y[va], score(m, sc.transform(X[va]), how)))
            J = 0.5 * (np.mean(aa) + np.min(aa))
            if best is None or J > best[0]: best = (J, p)
        sc = StandardScaler().fit(X); m = clone(est0).set_params(**best[1]).fit(sc.transform(X), y)
        return score(m, sc.transform(Xt), how), {k.split("__")[-1]: (str(v) if isinstance(v, tuple) else v) for k, v in best[1].items()}
    return run


def specs_small(seed):
    """Reduced grids for the event-grouped ISO-NE protocol (centred on the synthetic selections,
    including the widened edges found by grid_ext.py)."""
    full = specs(seed)
    grids = {"Poly-SVM": {"C": [0.1, 1, 10], "degree": [2, 3]},
             "Laplacian-SVM": {"C": [0.1, 1, 10], "gamma": [1/8, 1/4, 1/2, 1]},
             "MLP": {"hidden_layer_sizes": [(32,)], "alpha": [0.1, 1, 10]},
             "GBoost": {"learning_rate": [0.02, 0.05], "max_depth": [1, 2], "max_iter": [100, 300]},
             "RFF-256": {"rff__gamma": [0.05, 0.1, 0.5], "lr__C": [0.1, 1, 10]},
             "CosProd-SVM": {"C": [0.1, 1, 10], "omega": [0.5, 1.0, 2.0]}}
    return {k: (full[k][0], grids[k], full[k][2]) for k in grids}
