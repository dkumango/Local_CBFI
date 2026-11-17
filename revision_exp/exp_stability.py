"""
(1) Convergence / CV of localized CBFI v2 vs n_samples (Figs. 9-10), 6 datasets x 6 models.
(2) E4: sampling-strategy sensitivity (marginal vs conditional KNN) — Reviewer 1, 2.5.

Outputs: results/stability_convergence.csv, results/stability_sampler.csv
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import split, fit_model, background, save, MODELS
import local_cbfi_v2 as V2

DATASETS = ["adult", "bank", "diabetes", "california", "bike", "ames"]
NS = [10, 25, 50, 100, 200, 500]
N_INST, N_REP = 5, 30


def convergence(ds, mname):
    try:
        Xtr, Xte, ytr, yte, task = split(ds)
        model = fit_model(mname, task, Xtr, ytr)
        bg = background(Xtr)
        rows = []
        for i in range(N_INST):
            x = Xte.iloc[i]
            # target feature: largest |Imp| at a high-precision reference run (n = 1000)
            ref = V2.decompose_instance(model, x, bg, task=task, n_samples=1000, rng=10_000 + i)
            feat = ref.index[0]
            for n in NS:
                vals = [V2.decompose_feature(model, x, bg, feat, task=task, n_samples=n, rng=r) for r in range(N_REP)]
                g4 = np.array([v["G4"] for v in vals]); g1 = np.array([v["G1"] for v in vals])
                imp = np.array([v["Imp"] for v in vals])
                rows.append({"dataset": ds, "model": mname, "instance": i, "feature": feat, "n": n,
                             "G4_mean": g4.mean(), "G4_sd": g4.std(), "G1_mean": g1.mean(), "G1_sd": g1.std(),
                             "Imp_mean": imp.mean(), "Imp_sd": imp.std(),
                             "G4_ref": ref.loc[feat, "G4"], "Imp_ref": ref.loc[feat, "Imp"],
                             # CV relative to |Imp| (G4 itself can be ~0, making its own CV meaningless)
                             "cv_G4_rel_Imp": g4.std() / (abs(ref.loc[feat, "Imp"]) + 1e-12),
                             "cv_Imp": imp.std() / (abs(imp.mean()) + 1e-12)})
        print(f"conv done {ds} {mname}", flush=True)
        return rows
    except Exception as e:
        print(f"FAILED conv {ds} {mname}: {e!r}", flush=True)
        return []


SAMPLERS = [("marginal", {}), ("conditional", {"n_neighbors": 20}),
            ("conditional", {"n_neighbors": 50}), ("conditional", {"n_neighbors": 200})]


def sampler_sensitivity(ds, n_inst=20):
    Xtr, Xte, ytr, yte, task = split(ds)
    model = fit_model("RF", task, Xtr, ytr)
    bg = background(Xtr)
    rows = []
    for i in range(n_inst):
        x = Xte.iloc[i]
        res = {}
        for s, kw in SAMPLERS:
            tag = s if not kw else f"{s}-k{kw['n_neighbors']}"
            res[tag] = V2.decompose_instance(model, x, bg, task=task, n_samples=200, sampler=s, rng=i, **kw).loc[x.index]
        base = res["marginal"]
        for tag, r in res.items():
            top3 = lambda d, c: set(d[c].abs().sort_values(ascending=False).index[:3])
            rows.append({"dataset": ds, "instance": i, "sampler": tag,
                         "kendall_Imp_vs_marginal": kendalltau(base.Imp.abs(), r.Imp.abs())[0],
                         "kendall_G4_vs_marginal": kendalltau(base.G4_abs, r.G4_abs)[0],
                         "top3_Imp_overlap": len(top3(base, "Imp") & top3(r, "Imp")) / 3,
                         "mean_abs_G4": r.G4_abs.mean(), "mean_abs_Imp": r.Imp.abs().mean(),
                         "G4_share": r.G4_abs.sum() / (r.G1.abs().sum() + r.G4_abs.sum() + 1e-12)})
    print(f"sampler done {ds}", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(d, m) for d in DATASETS for m in MODELS if m != "TabNet"]
    out = Parallel(n_jobs=10)(delayed(convergence)(*j) for j in jobs)
    out += [convergence(d, "TabNet") for d in DATASETS]
    save(pd.DataFrame([r for rows in out for r in rows]), "stability_convergence.csv")

    out = Parallel(n_jobs=6)(delayed(sampler_sensitivity)(d) for d in ["wine", "insurance", "california", "bike", "adult", "diabetes"])
    save(pd.DataFrame([r for rows in out for r in rows]), "stability_sampler.csv")
