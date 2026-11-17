"""
Robustness of the case taxonomy on fitted models (instance-level, bootstrap-based labels).

Part A — synthetic: ground-truth labels come from the true function (DGP, n = 2000 samples);
  labels on fitted models (RF, XGB, ANN; n = 200) are compared with them.
  Methods: per-sample majority case (previous approach) vs instance-level labels for kappa in KAPPAS.
Part B — real data (RF): label distribution, label stability over 5 re-runs, smoker x BMI.
Outputs: results/taxonomy_synthetic.csv (one row per instance x feature x level x seed x method),
         results/taxonomy_real.csv
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_cbfi_v2 as V2
from common import split, fit_model, background, save

P = 6
COLS = [f"x{i+1}" for i in range(P)]
KAPPAS = [0.0, 0.05, 0.1, 0.2]
N_INST, N_MODEL, N_DGP = 40, 200, 2000

SCEN = {
    "additive":  lambda X: X[:, 0] + X[:, 1] + X[:, 2],
    "pair":      lambda X: X[:, 0] + X[:, 1] + 5 * X[:, 2] * X[:, 3],
    "AND":       lambda X: ((X[:, 0] > .5) & (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "OR":        lambda X: ((X[:, 0] > .5) | (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "max":       lambda X: np.maximum(X[:, 0], X[:, 1]) + .5 * X[:, 2],
    "gate":      lambda X: 3 * X[:, 0] * (X[:, 1] > .5) + X[:, 2],
    "reverse":   lambda X: 2 * X[:, 0] - 3 * X[:, 0] * X[:, 1],
    "xor":       lambda X: ((X[:, 0] > .5) != (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "duplicate": lambda X: 2 * X[:, 0] + X[:, 2],          # x2 = x1 + noise; judged separately
}


def gen(n, scen, rng):
    X = rng.random((n, P))
    if scen == "duplicate":
        X[:, 1] = np.clip(X[:, 0] + rng.normal(0, 0.02, n), 0, 1)
    return X


class Fn:
    def __init__(self, f): self.f = f
    def predict(self, df): return self.f(np.asarray(df, dtype=float))


def fit(level, f, scen, seed, rng):
    if level == "DGP":
        return Fn(f)
    Xtr = gen(4000, scen, rng)
    ytr = f(Xtr) + rng.normal(0, .05, len(Xtr))
    df = pd.DataFrame(Xtr, columns=COLS)
    if level == "RF":
        return RandomForestRegressor(300, min_samples_leaf=3, n_jobs=1, random_state=seed).fit(df, ytr)
    if level == "XGB":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=seed).fit(df, ytr)
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=True, random_state=seed).fit(df, ytr)


TAUS = [0.01, 0.05, 0.1, 0.2]           # practical-relevance threshold, fraction of sd(f)


def settings():
    s = [(t, 0.1) for t in TAUS] + [(0.1, k) for k in KAPPAS if k != 0.1]
    return s


def labels_for(model, xs, bg, feat, n, rng, sd):
    r = V2.decompose_feature(model, xs, bg, feat, n_samples=n, rng=rng, return_samples=True)
    s = r["samples"]
    a, b, c = s.a.to_numpy(), s.b.to_numpy(), s.c.to_numpy()
    out = {"per-sample majority": s.case.value_counts().idxmax()}
    for t, k in settings():
        lab = V2.classify_effect(a, b, c, tau=t * sd, kappa=k, rng=rng)
        out[f"instance tau={t} k={k}"] = lab["label"]
        out[f"conf tau={t} k={k}"] = lab["label_conf"]
    return out


def run(scen, seed):
    rng = np.random.default_rng(seed)
    f = SCEN[scen]
    bg = pd.DataFrame(gen(2000, scen, rng), columns=COLS)
    Xi = gen(N_INST, scen, rng)
    models = {lv: fit(lv, f, scen, seed, rng) for lv in ["DGP", "RF", "XGB", "ANN"]}
    sd = float(np.std(models["DGP"].predict(bg.iloc[:500])))     # one output scale for all levels
    meths = ["per-sample majority"] + [f"instance tau={t} k={k}" for t, k in settings()]
    rows = []
    for k, x in enumerate(Xi):
        xs = pd.Series(x, index=COLS)
        for feat in COLS:
            gt = labels_for(models["DGP"], xs, bg, feat, N_DGP, seed * 10_000 + k, sd)
            for lv in ["RF", "XGB", "ANN"]:
                est = labels_for(models[lv], xs, bg, feat, N_MODEL, seed * 10_000 + k, sd)
                for m in meths:
                    g = gt[m] if m != "per-sample majority" else gt["instance tau=0.1 k=0.1"]
                    rows.append({"scenario": scen, "seed": seed, "instance": k, "feature": feat, "level": lv,
                                 "method": m, "gt": g, "pred": est[m],
                                 "conf": est.get(m.replace("instance", "conf"), np.nan)})
    print(f"done {scen} seed={seed}", flush=True)
    return rows


def run_real(ds, n_inst=30, reps=5):
    Xtr, Xte, ytr, yte, task = split(ds)
    model = fit_model("RF", task, Xtr, ytr)
    bg = background(Xtr)

    def one(i):
        x = Xte.iloc[i]
        out = []
        for feat in x.index:
            labs = [V2.decompose_feature(model, x, bg, feat, task=task, n_samples=200, rng=1000 * r + i)
                    for r in range(reps)]
            modal = pd.Series([l["label"] for l in labs]).value_counts()
            out.append({"dataset": ds, "instance": i, "feature": feat, "label": modal.index[0],
                        "rerun_agreement": modal.iloc[0] / reps,
                        "label_conf": np.mean([l["label_conf"] for l in labs]),
                        "G1": np.mean([l["G1"] for l in labs]), "G4": np.mean([l["G4"] for l in labs]),
                        "Imp": np.mean([l["Imp"] for l in labs]),
                        **({"smoker": x["smoker"], "bmi": x["bmi"]} if ds == "insurance" else {})})
        return out
    parts = Parallel(n_jobs=20)(delayed(one)(i) for i in range(n_inst))
    return [r for p in parts for r in p]


if __name__ == "__main__":
    out = Parallel(n_jobs=20)(delayed(run)(s, sd) for s in SCEN for sd in range(3))
    save(pd.DataFrame([r for rows in out for r in rows]), "taxonomy_synthetic.csv")
    real = []
    for ds in ["insurance", "wine", "california", "bike"]:
        real += run_real(ds)
    save(pd.DataFrame(real), "taxonomy_real.csv")
