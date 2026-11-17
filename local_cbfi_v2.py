"""
Localized CBFI v2 — case-based decomposition into standalone (G1) and interaction (G4) effects.

Revision (A') for the PeerJ CS resubmission. For every background sample z^(s) the explained
instance x is compared in four hybrid states built from x and z^(s):

    v_full   = f(x)                      all features from x
    v_minus  = f(x with feature i <- z)  feature i removed, the rest kept   (in context)
    v_alone  = f(z with feature i <- x)  only feature i kept                (standalone)
    v_none   = f(z)                      nothing kept

Per sample s:
    b_s = v_full  - v_minus   in-context effect of i  (local permutation effect)
    a_s = v_alone - v_none    standalone effect of i
    d_s = b_s - a_s           interaction effect: the part of i's effect that depends on the other
                              features being present (zero for every s when f is additive in i)

Each sample is routed to one case:
    inactive     |a_s| <= tau and |b_s| <= tau
    standalone   |d_s| <= tau                       (i acts the same alone and in context)      ~ CBFI G1
    reversed     |a_s|, |b_s| > tau, sign(a) != sign(b)
                                                    (context flips the direction of i: suppression)
    amplified    otherwise and |b_s| >  |a_s|       (context strengthens i's effect: synergy, AND) ~ CBFI G4
    redundant    otherwise, |b_s| <= |a_s| and the rest of x alone (c_s = v_minus - v_none) moves the
                 output in i's direction by at least half of the lost effect: c_s*sign(a_s) >= |d_s|/2
                                                    (substitution, OR)                          ~ CBFI G3
    gated        otherwise, |b_s| <= |a_s|          (the context switches i off, e.g. unmet AND)
"attenuated" = redundant + gated + reversed is kept as an aggregate for backward compatibility.

Aggregates over s = 1..n:
    G1  = mean(a_s)            standalone effect (signed, output units)
    G4  = mean(d_s)            interaction effect (signed, output units)
    Imp = mean(b_s) = G1 + G4  in-context effect
    plus the case shares, which say how often the interaction is active for this instance.

Output v is the regression prediction, or for classification the probability (or log-odds)
of the class predicted for x. Labels are not used.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

CASES = ("inactive", "standalone", "amplified", "redundant", "gated", "reversed")


def route_cases(a, b, c, tau):
    """
    Assign each sample to one case.
      a: standalone effect of i      f(x_i, z_-i) - f(z)
      b: in-context effect of i      f(x) - f(x_-i, z_i)
      c: effect of the rest without i  f(x_-i, z_i) - f(z)
    Weakened-but-same-direction samples (|b| <= |a|) are split by whether the rest of x alone already
    moves the output in i's direction (redundant / substitution, original CBFI G3) or not (gated:
    the context switches i off, e.g. an unmet AND condition).
    """
    d = b - a
    big_a, big_b = np.abs(a) > tau, np.abs(b) > tau
    return np.select(
        [~big_a & ~big_b,
         np.abs(d) <= tau,
         big_a & big_b & (np.sign(a) != np.sign(b)),
         np.abs(b) > np.abs(a),
         # the rest of x alone reproduces at least half of the effect i lost in context
         (c * np.sign(a) >= 0.5 * np.abs(d)) & (np.abs(c) > tau)],
        ["inactive", "standalone", "reversed", "amplified", "redundant"],
        default="gated")


# --------------------------------------------------------------------------------------
# Output function
# --------------------------------------------------------------------------------------
def make_value_fn(model, x, task="regression", output="proba"):
    """Return v(DataFrame) -> 1-D array, the quantity that is decomposed."""
    x_df = pd.DataFrame([x])
    if task in ("regression", "reg"):
        return lambda df: np.asarray(model.predict(df), dtype=float)

    proba_x = model.predict_proba(x_df)[0]
    c = int(np.argmax(proba_x))

    if output == "proba":
        return lambda df: model.predict_proba(df)[:, c]
    if output == "logit":
        def v(df):
            p = np.clip(model.predict_proba(df)[:, c], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        return v
    if output == "label":
        return lambda df: (np.argmax(model.predict_proba(df), axis=1) == c).astype(float)
    raise ValueError(output)


# --------------------------------------------------------------------------------------
# Background samples
# --------------------------------------------------------------------------------------
def draw_background(x, background, features_to_condition_on=None, n_samples=100,
                    sampler="marginal", n_neighbors=50, n_cond=2, rng=None):
    """
    marginal    : rows drawn uniformly from the background data.
    conditional : rows drawn from the k nearest background rows of x, measured on the n_cond
                  features most correlated with the target feature(s) (same idea as v1).
    """
    rng = np.random.default_rng(rng)
    if sampler == "marginal":
        idx = rng.integers(0, len(background), n_samples)
        return background.iloc[idx].reset_index(drop=True)

    if sampler == "conditional":
        targets = list(features_to_condition_on)
        corr = background.corr().abs()[targets].mean(axis=1).drop(targets, errors="ignore")
        cond = corr.sort_values(ascending=False).index[:n_cond].tolist()
        B = background[cond].to_numpy(dtype=float)
        mu, sd = B.mean(0), B.std(0) + 1e-12          # standardize so raw units do not dominate
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(background)))
        nn.fit((B - mu) / sd)
        q = (x[cond].to_numpy(dtype=float) - mu) / sd
        neigh = nn.kneighbors(q[None, :], return_distance=False)[0]
        idx = rng.choice(neigh, n_samples, replace=True)
        return background.iloc[idx].reset_index(drop=True)

    raise ValueError(sampler)


def _hybrid(z, x, keep):
    """Copy of z with the columns in `keep` set to the values of x."""
    h = z.copy()
    for k in keep:
        h[k] = x[k]
    return h


# --------------------------------------------------------------------------------------
# Single-feature decomposition
# --------------------------------------------------------------------------------------
def _route_means(A, B, C, D, tau, kappa):
    """Instance-level routing on (bootstrap) means; arrays of equal shape."""
    big_a, big_b = np.abs(A) > tau, np.abs(B) > tau
    small_d = np.abs(D) <= np.maximum(tau, kappa * np.maximum(np.abs(A), np.abs(B)))
    return np.select(
        [~big_a & ~big_b,
         small_d,
         big_a & big_b & (np.sign(A) != np.sign(B)),
         np.abs(B) > np.abs(A),
         C * np.sign(A) >= 0.5 * np.abs(D)],
        ["inactive", "standalone", "reversed", "amplified", "redundant"],
        default="gated")


def classify_effect(a, b, c, tau, kappa=0.1, n_boot=300, alpha=0.05, rng=None):
    """
    One robust label per (instance, feature).
    The per-sample effects are averaged first, so the jaggedness of a fitted model (which makes single
    samples look interactive) cancels out; an interaction label additionally requires the interaction
    to be practically relevant: |mean d| > kappa * max(|mean a|, |mean b|).
    Uncertainty: the samples are bootstrapped, the rule is applied to every replicate, and the modal
    label is returned with its bootstrap frequency (label_conf) and a CI for mean d.
    """
    seed = [int(rng), 7919] if isinstance(rng, (int, np.integer)) else None   # reproducible, independent stream
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    A, B, C = a[idx].mean(1), b[idx].mean(1), c[idx].mean(1)
    D = B - A
    labels = _route_means(A, B, C, D, tau, kappa)
    vals, cnt = np.unique(labels, return_counts=True)
    point = str(_route_means(a.mean(), b.mean(), c.mean(), b.mean() - a.mean(), tau, kappa))
    mode = str(vals[np.argmax(cnt)])
    q = lambda v: np.quantile(v, [alpha / 2, 1 - alpha / 2])
    (lo, hi), (alo, ahi), (blo, bhi) = q(D), q(A), q(B)
    covers0 = lambda l, h: l <= 0 <= h
    conf = float(cnt.max() / n_boot)
    # statistical guards (fitted models give irrelevant features tiny, noisy, non-zero effects):
    if covers0(alo, ahi) and covers0(blo, bhi):
        mode, conf = "inactive", float(np.mean(labels == "inactive"))   # no detectable effect at all
    elif mode in ("amplified", "redundant", "gated", "reversed") and covers0(lo, hi):
        mode, conf = "standalone", float(np.mean(labels == "standalone"))  # interaction not significant
    return {"label": mode, "label_conf": conf, "label_point": point,
            "G4_lo": float(lo), "G4_hi": float(hi)}


def decompose_feature(model, x, background, feature, task="regression", output="proba",
                      n_samples=100, sampler="marginal", tau=None, rng=None, return_samples=False,
                      kappa=0.1, tau_label=0.1, n_boot=300, **sampler_kw):
    """tau (default 1% of sd of f over the background) is used for the per-sample case shares;
    tau_label * sd is the practical-relevance threshold of the instance-level label (default 10%)."""
    v = make_value_fn(model, x, task, output)
    z = draw_background(x, background, [feature], n_samples, sampler, rng=rng, **sampler_kw)
    others = [c for c in x.index if c != feature]

    v_full = float(v(pd.DataFrame([x]))[0])
    v_minus = v(_hybrid(z, x, others))
    v_alone = v(_hybrid(z, x, [feature]))
    v_none = v(z)

    b = v_full - v_minus
    a = v_alone - v_none
    d = b - a

    if tau is None:
        tau = 0.01 * (np.std(v_none) + 1e-12)

    rest = v_minus - v_none
    case = route_cases(a, b, rest, tau)
    lab = classify_effect(a, b, rest, tau=tau_label * (np.std(v_none) + 1e-12), kappa=kappa,
                          n_boot=n_boot, rng=rng)

    out = {"G1": a.mean(), "G4": d.mean(), "Imp": b.mean(),
           "G4_abs": np.abs(d).mean()}
    for k in CASES:
        out[f"share_{k}"] = float(np.mean(case == k))
    out["share_attenuated"] = out["share_redundant"] + out["share_gated"] + out["share_reversed"]
    # interaction mass by case (sum of d within each case, divided by n)
    for k in ("amplified", "redundant", "gated", "reversed"):
        out[f"G4_{k}"] = float(np.sum(d[case == k]) / len(d))
    out.update(lab)
    if return_samples:
        out["samples"] = pd.DataFrame({"a": a, "b": b, "c": rest, "d": d, "case": case})
    return out


def decompose_instance(model, x, background, task="regression", **kw):
    rows = {f: decompose_feature(model, x, background, f, task=task, **kw) for f in x.index}
    df = pd.DataFrame(rows).T
    return df.sort_values("Imp", key=np.abs, ascending=False)


# --------------------------------------------------------------------------------------
# Pairwise interaction
# --------------------------------------------------------------------------------------
def pairwise_interaction(model, x, background, fi, fj, task="regression", output="proba",
                         n_samples=100, sampler="marginal", rng=None, **sampler_kw):
    """
    G4_ij = mean_s [ f(x) - f(x_{-i} z_i) - f(x_{-j} z_j) + f(x_{-ij} z_ij) ]
          = (effect of i when j is present) - (effect of i when j is removed), others kept at x.
    Symmetric in i, j; zero for every s when f has no i-j interaction term.
    """
    v = make_value_fn(model, x, task, output)
    z = draw_background(x, background, [fi, fj], n_samples, sampler, rng=rng, **sampler_kw)
    cols = list(x.index)
    v_full = float(v(pd.DataFrame([x]))[0])
    v_i = v(_hybrid(z, x, [c for c in cols if c != fi]))
    v_j = v(_hybrid(z, x, [c for c in cols if c != fj]))
    v_ij = v(_hybrid(z, x, [c for c in cols if c not in (fi, fj)]))
    d = v_full - v_i - v_j + v_ij
    return {"G4_ij": d.mean(), "G4_ij_abs": np.abs(d).mean()}


def interaction_matrix(model, x, background, features=None, **kw):
    features = list(features or x.index)
    M = pd.DataFrame(0.0, index=features, columns=features)
    for p, fi in enumerate(features):
        for fj in features[p + 1:]:
            val = pairwise_interaction(model, x, background, fi, fj, **kw)["G4_ij"]
            M.loc[fi, fj] = M.loc[fj, fi] = val
    return M
