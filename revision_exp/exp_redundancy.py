"""
Validation of the refined case routing (standalone / amplified / redundant / gated / reversed).

Part A — synthetic functions with known local logic (DGP = true function; RF = fitted model).
  Expected cases for feature x1, by instance subgroup:
    additive  x1+x2+x3                      -> standalone
    AND       1[x1>.5 & x2>.5] (+.5 x3)    x1 hi & x2 hi -> amplified ; x1 hi & x2 lo -> gated
    OR        1[x1>.5 | x2>.5] (+.5 x3)    x1 hi & x2 hi -> redundant ; x1 hi & x2 lo -> amplified
    reverse   2 x1 - 3 x1 x2                x2 > 2/3 -> reversed ; x2 < 1/3 -> amplified/standalone
    duplicate y = 2 x1 + x3, x2 = x1 + noise (model level only) -> x1, x2 redundant for RF
Part B — Wine and Insurance (RF): case shares per feature, and their relation to how strongly a feature
  is correlated with the others (redundancy should rise with correlation).
Outputs: results/redundancy_synthetic.csv, results/redundancy_real.csv
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_cbfi_v2 as V2
from common import split, fit_model, background, save

P = 6
COLS = [f"x{i+1}" for i in range(P)]
SHARES = [f"share_{c}" for c in V2.CASES]

SCEN = {
    "additive":  lambda X: X[:, 0] + X[:, 1] + X[:, 2],
    "AND":       lambda X: ((X[:, 0] > .5) & (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "OR":        lambda X: ((X[:, 0] > .5) | (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "reverse":   lambda X: 2 * X[:, 0] - 3 * X[:, 0] * X[:, 1],
    "duplicate": lambda X: 2 * X[:, 0] + X[:, 2],
}


def gen(n, scen, rng):
    X = rng.random((n, P))
    if scen == "duplicate":
        X[:, 1] = np.clip(X[:, 0] + rng.normal(0, 0.02, n), 0, 1)
    return X


def subgroup(scen, x):
    hi1, hi2 = x[0] > .5, x[1] > .5
    if scen in ("AND", "OR"):
        if hi1 and hi2: return "x1 hi, x2 hi"
        if hi1 and not hi2: return "x1 hi, x2 lo"
        return "x1 lo"
    if scen == "reverse":
        return "x2 > 2/3" if x[1] > 2 / 3 else ("x2 < 1/3" if x[1] < 1 / 3 else "x2 mid")
    return "all"


class Fn:
    def __init__(self, f): self.f = f
    def predict(self, df): return self.f(np.asarray(df, dtype=float))


def run_syn(scen, level, seed, n_inst=60, n_samples=300):
    rng = np.random.default_rng(seed)
    f = SCEN[scen]
    if level == "DGP":
        model = Fn(f)
    else:
        Xtr = gen(4000, scen, rng)
        model = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=1, random_state=seed).fit(
            pd.DataFrame(Xtr, columns=COLS), f(Xtr) + rng.normal(0, .05, len(Xtr)))
    bg = pd.DataFrame(gen(2000, scen, rng), columns=COLS)
    rows = []
    for k, x in enumerate(gen(n_inst, scen, rng)):
        xs = pd.Series(x, index=COLS)
        for feat in ["x1", "x2"]:
            r = V2.decompose_feature(model, xs, bg, feat, n_samples=n_samples, rng=seed * 1000 + k)
            rows.append({"scenario": scen, "level": level, "seed": seed, "instance": k, "feature": feat,
                         "subgroup": subgroup(scen, x), **{s: r[s] for s in SHARES},
                         "G1": r["G1"], "G4": r["G4"], "Imp": r["Imp"]})
    return rows


def run_real(ds, n_inst=50):
    Xtr, Xte, ytr, yte, task = split(ds)
    model = fit_model("RF", task, Xtr, ytr)
    bg = background(Xtr)
    corr = Xtr.corr().abs()
    np.fill_diagonal(corr.values, 0)

    def one(i):
        x = Xte.iloc[i]
        out = []
        for feat in x.index:
            r = V2.decompose_feature(model, x, bg, feat, task=task, n_samples=200, rng=i)
            out.append({"dataset": ds, "instance": i, "feature": feat, **{s: r[s] for s in SHARES},
                        "G1": r["G1"], "G4": r["G4"], "Imp": r["Imp"],
                        **{f"G4_{c}": r[f"G4_{c}"] for c in ("amplified", "redundant", "gated", "reversed")},
                        "max_abs_corr": corr.loc[feat].max()})
        return out
    parts = Parallel(n_jobs=20)(delayed(one)(i) for i in range(n_inst))
    return [r for p in parts for r in p]


if __name__ == "__main__":
    jobs = [(s, l, sd) for s in SCEN for l in ["DGP", "RF"] for sd in range(3)]
    out = Parallel(n_jobs=20)(delayed(run_syn)(*j) for j in jobs)
    save(pd.DataFrame([r for rows in out for r in rows]), "redundancy_synthetic.csv")
    save(pd.DataFrame(run_real("wine") + run_real("insurance")), "redundancy_real.csv")
