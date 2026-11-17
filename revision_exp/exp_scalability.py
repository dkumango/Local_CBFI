"""
E3: scalability to higher dimensions (Reviewer 2, Exp-2).
y = sum of 5 pairwise products among the first 10 features + linear terms of 5 further features; the rest is noise.
p in {10, 25, 50, 100, 200}; RF and XGB.
Methods: CBFI full pairwise (p <= 50), CBFI two-stage (screen top-m by mean|d|, pairwise among them),
         TreeSHAP interaction values (exact, tree-specific), shapiq k-SII (approximate, budget 2000).
Metric: recall of the 5 true pairs among the top-5 pairs; runtime per instance.
Output: results/scalability.csv
"""
import os, sys, time, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_cbfi_v2 as V2
from common import save

TRUE = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
PS, N_INST, N_SAMPLES, M_SCREEN = [10, 25, 50, 100, 200], 5, 100, 12


def gen(n, p, rng):
    X = rng.random((n, p))
    y = sum(4 * X[:, i] * X[:, j] for i, j in TRUE) + sum(X[:, k] for k in range(10, 15) if k < p)
    return X, y + rng.normal(0, 0.05, n)


def recall_top(scores, k=5):
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return len(set(top) & set(TRUE)) / len(TRUE)


def run(p, level, seed):
    rng = np.random.default_rng(seed)
    cols = [f"x{i+1}" for i in range(p)]
    Xtr, ytr = gen(5000, p, rng)
    df = pd.DataFrame(Xtr, columns=cols)
    if level == "RF":
        model = RandomForestRegressor(300, max_features=0.5, min_samples_leaf=3, n_jobs=1, random_state=seed).fit(df, ytr)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=seed).fit(df, ytr)
    bg = pd.DataFrame(gen(1000, p, rng)[0], columns=cols)
    Xinst = gen(N_INST, p, rng)[0]
    rows = []

    def add(method, rec, sec, **kw):
        rows.append({"p": p, "level": level, "seed": seed, "method": method, "recall@5": rec, "sec_per_instance": sec, **kw})

    for k, x in enumerate(Xinst):
        xs = pd.Series(x, index=cols)
        # CBFI two-stage
        t = time.perf_counter()
        scr = {f: V2.decompose_feature(model, xs, bg, f, n_samples=N_SAMPLES, rng=k)["G4_abs"] for f in cols}
        top = sorted(scr, key=scr.get, reverse=True)[:M_SCREEN]
        sc = {}
        for a, b in itertools.combinations(top, 2):
            ia, ib = sorted((cols.index(a), cols.index(b)))
            sc[(ia, ib)] = abs(V2.pairwise_interaction(model, xs, bg, a, b, n_samples=N_SAMPLES, rng=k)["G4_ij"])
        add("CBFI two-stage", recall_top(sc), time.perf_counter() - t,
            screen_recall=len({c for pr in TRUE for c in pr} & {cols.index(f) for f in top}) / 10)
        # CBFI full pairwise
        if p <= 50:
            t = time.perf_counter()
            sc = {(i, j): abs(V2.pairwise_interaction(model, xs, bg, cols[i], cols[j], n_samples=N_SAMPLES, rng=k)["G4_ij"])
                  for i, j in itertools.combinations(range(p), 2)}
            add("CBFI full", recall_top(sc), time.perf_counter() - t)
        # shapiq k-SII (approximate)
        if p <= 100:
            import shapiq
            f = lambda A: model.predict(pd.DataFrame(A, columns=cols))
            t = time.perf_counter()
            ex = shapiq.TabularExplainer(model=f, data=bg.sample(100, random_state=k).to_numpy(), index="k-SII", max_order=2, random_state=k)
            M = np.asarray(ex.explain(x, budget=2000).get_n_order_values(2))
            add("kSII (budget 2000)", recall_top({(i, j): abs(M[i, j]) for i, j in itertools.combinations(range(p), 2)}),
                time.perf_counter() - t)
    # TreeSHAP interaction values (all instances at once)
    import shap
    t = time.perf_counter()
    siv = shap.TreeExplainer(model).shap_interaction_values(pd.DataFrame(Xinst, columns=cols))
    sec = (time.perf_counter() - t) / N_INST
    for S in siv:
        add("TreeSHAP-IQ", recall_top({(i, j): abs(S[i, j]) for i, j in itertools.combinations(range(p), 2)}), sec)
    print(f"done p={p} {level} seed={seed}", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(p, l, s) for p in PS for l in ["RF", "XGB"] for s in range(3)]
    out = Parallel(n_jobs=15)(delayed(run)(*j) for j in jobs)
    save(pd.DataFrame([r for rows in out for r in rows]), "scalability.csv")
