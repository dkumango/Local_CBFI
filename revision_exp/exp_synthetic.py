"""
E1 + E2 (+ part of E7): synthetic interaction-recovery benchmark under identical conditions.

Methods (pairwise interaction scores per instance, 8 x 8):
  CBFI      : |G4_ij| from local_cbfi_v2 (marginal sampler; also conditional for the 'correlated' scenario)
  kSII      : shapiq k-SII, order 2, marginal imputer, exact (budget 2^p)          [model-agnostic]
  TreeSHAPIQ: shap TreeExplainer interaction values (RF/XGB only, exact)            [tree-specific]
  H         : Friedman's pairwise H^2 (global only)
Levels (Reviewer 1, 2.4): DGP (true function as model), RF, XGB, ANN.
Outputs: results/synthetic_runs.csv (one row per run x method), results/synthetic_stability.csv
"""
import itertools, time, sys, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from scipy.stats import norm, kendalltau
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import local_cbfi_v2 as V2
from common import save

P = 8
COLS = [f"x{i+1}" for i in range(P)]
PAIRS = list(itertools.combinations(range(P), 2))
N_TRAIN, N_INST, N_SAMPLES, N_BG_SHAPIQ, SIGMA = 3000, 20, 200, 100, 0.05
SEEDS = range(5)
LEVELS = ["DGP", "RF", "XGB", "ANN"]

SCEN = {   # name: (f(X ndarray), true pairs, analytic pair coefficients for local GT or None)
    "additive":    (lambda X: X[:, 0] + 2 * X[:, 1] + X[:, 2] - X[:, 3], set(), {}),
    "pair":        (lambda X: X[:, 0] + X[:, 1] + 5 * X[:, 2] * X[:, 3], {(2, 3)}, {(2, 3): 5}),
    "suppressor":  (lambda X: 2 * X[:, 0] - 5 * X[:, 0] * X[:, 1] + X[:, 2], {(0, 1)}, {(0, 1): -5}),
    "two_pairs":   (lambda X: 3 * X[:, 0] * X[:, 1] - 3 * X[:, 2] * X[:, 3] + X[:, 4],
                    {(0, 1), (2, 3)}, {(0, 1): 3, (2, 3): -3}),
    "xor":         (lambda X: ((X[:, 0] > .5) != (X[:, 1] > .5)).astype(float) + 0.5 * X[:, 2], {(0, 1)}, None),
    "threeway":    (lambda X: 10 * X[:, 0] * X[:, 1] * X[:, 2], {(0, 1), (0, 2), (1, 2)}, None),
    "conditional": (lambda X: np.where(X[:, 4] > .5, 4 * X[:, 0] * X[:, 1], X[:, 0]), {(0, 1), (0, 4), (1, 4)}, None),
    "correlated":  (lambda X: X[:, 0] + X[:, 1] + 5 * X[:, 2] * X[:, 3], {(2, 3)}, None),   # x1 ~ x3, rho=.8
}


def gen_X(n, scen, rng):
    if scen != "correlated":
        return rng.random((n, P))
    C = np.eye(P); C[0, 2] = C[2, 0] = 0.8
    Z = rng.multivariate_normal(np.zeros(P), C, size=n)
    return norm.cdf(Z)


class Fn:
    def __init__(self, f): self.f = f
    def predict(self, df): return self.f(np.asarray(df, dtype=float))


def fit_level(level, f, Xtr, ytr, seed):
    df = pd.DataFrame(Xtr, columns=COLS)
    if level == "DGP":
        return Fn(f)
    if level == "RF":
        return RandomForestRegressor(300, min_samples_leaf=3, n_jobs=1, random_state=seed).fit(df, ytr)
    if level == "XGB":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=seed).fit(df, ytr)
    if level == "ANN":
        return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=True, random_state=seed).fit(df, ytr)


def friedman_h2(model, X, m=150, rng=None):
    """Pairwise H^2 (Friedman & Popescu 2008) on m sample points, centred partial dependences."""
    rng = np.random.default_rng(rng)
    S = X[rng.choice(len(X), m, replace=False)]
    pred = lambda A: model.predict(pd.DataFrame(A, columns=COLS))

    def pd_at(feats):
        out = np.empty(m)
        for r in range(m):
            A = S.copy(); A[:, feats] = S[r, feats]
            out[r] = pred(A).mean()
        return out - out.mean()

    pd1 = {j: pd_at([j]) for j in range(P)}
    H = np.zeros((P, P))
    for j, k in PAIRS:
        pjk = pd_at([j, k])
        H[j, k] = H[k, j] = np.sum((pjk - pd1[j] - pd1[k]) ** 2) / (np.sum(pjk ** 2) + 1e-12)
    return H


def pair_vector(M):
    return np.array([abs(M[i, j]) for i, j in PAIRS])


def metrics_from_scores(score_list, true_pairs, pred_sd):
    """score_list: list of 8x8 matrices (one per instance). Returns local + global metrics."""
    y = np.array([p in true_pairs for p in PAIRS], dtype=int)
    k = max(1, len(true_pairs))
    vecs = [pair_vector(M) for M in score_list]
    out = {}
    if 0 < y.sum() < len(y):
        out["local_auroc"] = np.mean([roc_auc_score(y, v) for v in vecs])
        out["local_p@k"] = np.mean([y[np.argsort(-v)[:k]].mean() for v in vecs])
        g = np.mean(vecs, axis=0)
        out["global_auroc"] = roc_auc_score(y, g)
        out["global_p@k"] = y[np.argsort(-g)[:k]].mean()
    null = [v[y == 0].max() for v in vecs]
    out["max_null_pair/sd(f)"] = np.mean(null) / pred_sd
    if y.sum():
        out["mean_true_pair/sd(f)"] = np.mean([v[y == 1].mean() for v in vecs]) / pred_sd
    return out


def explain_all(model, level, Xbg_df, Xinst, scen, seed):
    """Return dict method -> (list of matrices, seconds per instance)."""
    import shapiq
    res = {}
    pred_np = lambda A: model.predict(pd.DataFrame(A, columns=COLS))

    samplers = ["marginal", "conditional"] if scen == "correlated" else ["marginal"]
    for smp in samplers:
        mats, t0 = [], time.perf_counter()
        for k, x in enumerate(Xinst):
            xs = pd.Series(x, index=COLS)
            M = V2.interaction_matrix(model, xs, Xbg_df, n_samples=N_SAMPLES, sampler=smp,
                                      rng=seed * 1000 + k, **({"n_neighbors": 100} if smp == "conditional" else {}))
            mats.append(M.to_numpy())
        res["CBFI" if smp == "marginal" else "CBFI-cond"] = (mats, (time.perf_counter() - t0) / len(Xinst))

    bg = Xbg_df.sample(N_BG_SHAPIQ, random_state=seed).to_numpy()
    ex = shapiq.TabularExplainer(model=pred_np, data=bg, index="k-SII", max_order=2, random_state=seed)
    mats, t0 = [], time.perf_counter()
    for x in Xinst:
        iv = ex.explain(x, budget=2 ** P)
        mats.append(np.asarray(iv.get_n_order_values(2)))
    res["kSII"] = (mats, (time.perf_counter() - t0) / len(Xinst))

    if level in ("RF", "XGB"):
        import shap
        te = shap.TreeExplainer(model)
        t0 = time.perf_counter()
        siv = te.shap_interaction_values(pd.DataFrame(Xinst, columns=COLS))
        res["TreeSHAP-IQ"] = ([2 * s for s in siv], (time.perf_counter() - t0) / len(Xinst))
    return res


def one_run(scen, level, seed):
    f, true_pairs, coefs = SCEN[scen]
    rng = np.random.default_rng(seed)
    Xtr = gen_X(N_TRAIN, scen, rng)
    ytr = f(Xtr) + rng.normal(0, SIGMA, N_TRAIN)
    model = fit_level(level, f, Xtr, ytr, seed)
    Xbg = gen_X(2000, scen, rng)
    Xbg_df = pd.DataFrame(Xbg, columns=COLS)
    Xinst = gen_X(N_INST, scen, rng)
    pred_sd = float(np.std(model.predict(Xbg_df)))
    r2 = 1 - np.mean((model.predict(pd.DataFrame(Xtr, columns=COLS)) - ytr) ** 2) / np.var(ytr) if level != "DGP" else 1.0

    rows = []
    for meth, (mats, sec) in explain_all(model, level, Xbg_df, Xinst, scen, seed).items():
        row = {"scenario": scen, "level": level, "seed": seed, "method": meth, "sec_per_instance": sec,
               "train_r2": r2, **metrics_from_scores(mats, true_pairs, pred_sd)}
        if coefs:   # local analytic GT (independent uniform features, mean ~ 0.5)
            est, gt = [], []
            for M, x in zip(mats, Xinst):
                for (i, j) in PAIRS:
                    est.append(M[i, j]); gt.append(sum(c * (x[a] - .5) * (x[b] - .5) for (a, b), c in coefs.items() if (a, b) == (i, j)))
            est, gt = np.array(est), np.array(gt)
            row["corr_with_analytic"] = np.corrcoef(est, gt)[0, 1]
            mask = np.abs(gt) > .05
            row["sign_agree"] = np.mean(np.sign(est[mask]) == np.sign(gt[mask])) if mask.any() else np.nan
        rows.append(row)

    t0 = time.perf_counter()
    H = friedman_h2(model, Xbg, rng=seed)
    rows.append({"scenario": scen, "level": level, "seed": seed, "method": "H-stat (global)",
                 "sec_per_instance": np.nan, "sec_global": time.perf_counter() - t0, "train_r2": r2,
                 **{k: v for k, v in metrics_from_scores([H], true_pairs, 1.0).items() if k.startswith("global")}})
    print(f"done {scen:12s} {level:4s} seed={seed}", flush=True)
    return rows


def stability(n_inst=10, n_rep=10):
    """Explainer randomness only: fixed data/model (two_pairs, RF), vary the background/sampling seed."""
    f, true_pairs, _ = SCEN["two_pairs"]
    rng = np.random.default_rng(123)
    Xtr = rng.random((N_TRAIN, P)); ytr = f(Xtr) + rng.normal(0, SIGMA, N_TRAIN)
    model = fit_level("RF", f, Xtr, ytr, 0)
    Xpool = pd.DataFrame(rng.random((5000, P)), columns=COLS)
    Xinst = rng.random((n_inst, P))
    store = {}
    for r in range(n_rep):
        Xbg_df = Xpool.sample(2000, random_state=r).reset_index(drop=True)
        for meth, (mats, _) in explain_all(model, "RF", Xbg_df, Xinst, "two_pairs", seed=100 + r).items():
            store.setdefault(meth, []).append(np.stack([pair_vector(M) for M in mats]))   # inst x pairs
    rows = []
    tp = [PAIRS.index(p) for p in true_pairs]
    for meth, reps in store.items():
        A = np.stack(reps)                                    # rep x inst x pairs
        cv = (A[:, :, tp].std(0) / (A[:, :, tp].mean(0) + 1e-12)).mean()
        taus = [kendalltau(A[a, i], A[b, i])[0] for i in range(n_inst)
                for a, b in itertools.combinations(range(n_rep), 2)]
        rows.append({"method": meth, "cv_true_pairs": cv, "kendall_tau_pair_ranking": np.nanmean(taus)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    jobs = [(s, l, sd) for s in SCEN for l in LEVELS for sd in SEEDS]
    out = Parallel(n_jobs=int(os.environ.get("NJOBS", 20)), verbose=0)(delayed(one_run)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    save(df, "synthetic_runs.csv")
    summ = df.groupby(["scenario", "level", "method"]).mean(numeric_only=True).drop(columns="seed")
    summ.reset_index().pipe(save, "synthetic_summary.csv")
    save(stability(), "synthetic_stability.csv")
