"""
Sanity validation of local_cbfi_v2 on functions with known structure.
Two levels (Reviewer 1, 2.4):
  DGP   : the true function itself is the model  -> error of the explainer only
  MODEL : a RandomForest fitted to noisy samples -> explainer error + model approximation error
Analytic ground truth (uniform[0,1] features, marginal background, mean mu = 0.5):
  for a term c * x_i * x_j:  E[G4_i] gets c * (x_i - mu) * (x_j - mu); E[G4_ij] = same.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import local_cbfi_v2 as V2

P = 6
COLS = [f"x{i+1}" for i in range(P)]
rng = np.random.default_rng(42)
Xbg = pd.DataFrame(rng.random((2000, P)), columns=COLS)
MU = 0.5

SCEN = {
    # name: (f, true pairs {(i,j): coef})
    "additive":       (lambda d: d.x1 + 2 * d.x2 + d.x3,                         {}),
    "pair+additive":  (lambda d: d.x1 + d.x2 + 5 * d.x3 * d.x4,                  {("x3", "x4"): 5}),
    "suppressor":     (lambda d: 2 * d.x1 - 5 * d.x1 * d.x2,                     {("x1", "x2"): -5}),
    "two pairs":      (lambda d: 3 * d.x1 * d.x2 - 3 * d.x3 * d.x4 + d.x5,       {("x1", "x2"): 3, ("x3", "x4"): -3}),
    "xor":            (lambda d: ((d.x1 > .5) != (d.x2 > .5)).astype(float),     None),
    "3-way":          (lambda d: 10 * d.x1 * d.x2 * d.x3,                        None),
    "conditional":    (lambda d: np.where(d.x5 > .5, d.x1 * d.x2 * 4, d.x1),     None),
}
TRUE_INTERACTING = {"additive": set(), "pair+additive": {"x3", "x4"}, "suppressor": {"x1", "x2"},
                    "two pairs": {"x1", "x2", "x3", "x4"}, "xor": {"x1", "x2"},
                    "3-way": {"x1", "x2", "x3"}, "conditional": {"x1", "x2", "x5"}}
TRUE_ACTIVE = {"additive": {"x1", "x2", "x3"}, "pair+additive": {"x1", "x2", "x3", "x4"},
               "suppressor": {"x1", "x2"}, "two pairs": {"x1", "x2", "x3", "x4", "x5"},
               "xor": {"x1", "x2"}, "3-way": {"x1", "x2", "x3"}, "conditional": {"x1", "x2", "x5"}}


class Fn:
    def __init__(self, f): self.f = f
    def predict(self, df): return np.asarray(self.f(df), dtype=float)


def analytic_g4(x, pairs, feat):
    if pairs is None: return np.nan
    return sum(c * (x[i] - MU) * (x[j] - MU) for (i, j), c in pairs.items() if feat in (i, j))


def run(level, n_inst=30, n_samples=300):
    rows = []
    for name, (f, pairs) in SCEN.items():
        if level == "DGP":
            model = Fn(f)
        else:
            Xtr = pd.DataFrame(rng.random((3000, P)), columns=COLS)
            ytr = f(Xtr) + rng.normal(0, 0.05, len(Xtr))
            model = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        est, gt, g4share_add, g4share_int = [], [], [], []
        pair_hits = []
        for k in range(n_inst):
            x = Xbg.iloc[1000 + k]
            res = V2.decompose_instance(model, x, Xbg, n_samples=n_samples, rng=k)
            for feat in COLS:
                r = res.loc[feat]
                share = abs(r.G4_abs) / (abs(r.G1) + abs(r.G4_abs) + 1e-12)
                if feat in TRUE_ACTIVE[name] and feat not in TRUE_INTERACTING[name]:
                    g4share_add.append(share)
                if feat in TRUE_INTERACTING[name]:
                    g4share_int.append(share)
                if pairs is not None and feat in TRUE_ACTIVE[name]:
                    est.append(r.G4); gt.append(analytic_g4(x, pairs, feat))
            if pairs:   # pairwise ranking: are the true pairs the top-|G4_ij| pairs?
                M = V2.interaction_matrix(model, x, Xbg, n_samples=n_samples, rng=k).abs()
                vals = [(M.loc[a, b], (a, b)) for i, a in enumerate(COLS) for b in COLS[i + 1:]]
                top = {p for _, p in sorted(vals, reverse=True)[:len(pairs)]}
                pair_hits.append(len(top & set(pairs)) / len(pairs))
        rows.append({
            "level": level, "scenario": name,
            "G4 share | additive feats": np.mean(g4share_add) if g4share_add else np.nan,
            "G4 share | interacting feats": np.mean(g4share_int) if g4share_int else np.nan,
            "corr(G4, analytic)": np.corrcoef(est, gt)[0, 1] if est and np.std(gt) > 0 else np.nan,
            "sign agree (|gt|>.05)": (lambda e, g: np.mean(np.sign(e[np.abs(g) > .05]) == np.sign(g[np.abs(g) > .05])))(
                np.array(est), np.array(gt)) if est and np.std(gt) > 0 else np.nan,
            "pair recall@k": np.mean(pair_hits) if pair_hits else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = pd.concat([run("DGP"), run("MODEL")])
    print(out.round(3).to_string(index=False))
    out.to_csv("validate_v2_synthetic_results.csv", index=False)
