"""SHAP / LIME baselines on the same value function as localized CBFI, plus rank-agreement metrics."""
import time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
import local_cbfi_v2 as V2

TREE = ("RF", "XGB", "CAT")


def value_fn_np(model, x, task, cols):
    v = V2.make_value_fn(model, x, task)
    return lambda A: v(pd.DataFrame(np.asarray(A, dtype=float), columns=cols))


def shap_permutation(model, task, x, bg, seed=0):
    """Model-agnostic SHAP on the same output as CBFI (prediction / prob of predicted class)."""
    import shap
    cols = list(x.index)
    f = value_fn_np(model, x, task, cols)
    p = len(cols)
    ex = shap.PermutationExplainer(f, shap.maskers.Independent(bg.to_numpy(), max_samples=100), seed=seed)
    sv = ex(x.to_numpy()[None, :], max_evals=max(500, 2 * p + 1), silent=True)
    return np.asarray(sv.values[0])


def shap_tree_runtime(model_name, model, Xinst, bg):
    """Exact TreeSHAP runtime (s/instance) for tree models; None otherwise."""
    if model_name not in TREE:
        return None
    import shap
    pipe = model
    est, scaler = pipe.named_steps["est"], pipe.named_steps["scale"]
    t0 = time.perf_counter()
    te = shap.TreeExplainer(est)
    te.shap_values(scaler.transform(Xinst))
    return (time.perf_counter() - t0) / len(Xinst)


def lime_weights(model, task, x, Xtr, num_samples=5000, seed=0):
    from lime.lime_tabular import LimeTabularExplainer
    cols = list(x.index)
    f = value_fn_np(model, x, task, cols)
    ex = LimeTabularExplainer(Xtr.to_numpy(), feature_names=cols, mode="regression", random_state=seed)
    e = ex.explain_instance(x.to_numpy(), f, num_features=len(cols), num_samples=num_samples)
    lbl = list(e.local_exp.keys())[0]
    w = np.zeros(len(cols))
    for j, val in e.local_exp[lbl]:
        w[j] = val
    return w


def rbo(r1, r2, p=0.9):
    """Rank-biased overlap of two rankings (lists of indices), extrapolated (Webber et al. 2010)."""
    k = min(len(r1), len(r2))
    s1, s2, ov, acc = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        a, b = r1[d - 1], r2[d - 1]
        if a == b:
            ov += 1
        else:
            ov += (a in s2) + (b in s1)
        s1.add(a); s2.add(b)
        acc += (ov / d) * p ** d
    return (1 - p) / p * acc + (ov / k) * p ** k


def agreement(a, b, k=3):
    """Agreement between two attribution vectors on |value| rankings."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    ra, rb = list(np.argsort(-a)), list(np.argsort(-b))
    return {"spearman": spearmanr(a, b)[0], "kendall": kendalltau(a, b)[0],
            f"top{k}_jaccard": len(set(ra[:k]) & set(rb[:k])) / len(set(ra[:k]) | set(rb[:k])),
            "rbo": rbo(ra, rb)}
