"""
Mechanism identification: can an explanation tell HOW feature x1 interacts at an instance?

Ground truth is semantic (from the function and the instance's subgroup), not from CBFI's own definition:
  additive  x1+x2+x3                         -> standalone
  AND       x1 hi, x2 hi                     -> amplified (synergy)
            x1 hi, x2 lo                     -> gated     (unmet AND condition switches x1 off)
  gate      3*x1*1[x2>.5] + x3, x1 extreme, x2 hi -> amplified ; x2 lo -> gated
  OR        x1 hi, x2 hi                     -> redundant (either feature suffices)
  max       max(x1,x2) + .5 x3, both > .75   -> redundant
  reverse   2 x1 - 3 x1 x2, x1 extreme, x2 > .8 -> reversed (context flips the direction of x1)

Methods
  CBFI            : instance-level label of x1 (kappa = 0.1); does NOT know which feature is the partner
  kSII (sign)     : first-order phi_1 and pair phi_12 -> standalone / amplified / reversed / "attenuated";
                    cannot split attenuated into redundant vs gated (scored as wrong on both)
  kSII (+partner) : as above, and splits attenuated by the sign of the partner's own effect phi_2
                    (same direction -> redundant, otherwise gated); uses the true partner (an advantage)
  TreeSHAP-IQ (sign / +partner): same rules on exact TreeSHAP interaction values (RF, XGB)
Levels: DGP (true function), RF, XGB, ANN; 3 seeds; 15 instances per subgroup.
Output: results/mechanism.csv
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_cbfi_v2 as V2
from common import save
from exp_taxonomy import fit, Fn, COLS, P

KAPPA, N_PER_GROUP, N_SAMPLES = 0.1, 15, 200

FUNCS = {
    "additive": lambda X: X[:, 0] + X[:, 1] + X[:, 2],
    "AND":      lambda X: ((X[:, 0] > .5) & (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "gate":     lambda X: 3 * X[:, 0] * (X[:, 1] > .5) + X[:, 2],
    "OR":       lambda X: ((X[:, 0] > .5) | (X[:, 1] > .5)).astype(float) + .5 * X[:, 2],
    "max":      lambda X: np.maximum(X[:, 0], X[:, 1]) + .5 * X[:, 2],
    "reverse":  lambda X: 2 * X[:, 0] - 3 * X[:, 0] * X[:, 1],
}
far = lambda v: abs(v - .5) > .25
GROUPS = {   # scenario -> list of (subgroup name, condition on x, ground-truth mechanism)
    "additive": [("all", lambda x: far(x[0]), "standalone")],
    "AND":      [("x1 hi, x2 hi", lambda x: x[0] > .5 and x[1] > .5, "amplified"),
                 ("x1 hi, x2 lo", lambda x: x[0] > .5 and x[1] < .5, "gated")],
    "gate":     [("x2 hi", lambda x: far(x[0]) and x[1] > .5, "amplified"),
                 ("x2 lo", lambda x: far(x[0]) and x[1] < .5, "gated")],
    "OR":       [("x1 hi, x2 hi", lambda x: x[0] > .5 and x[1] > .5, "redundant")],
    "max":      [("both > .75", lambda x: x[0] > .75 and x[1] > .75, "redundant")],
    "reverse":  [("x2 > .8", lambda x: far(x[0]) and x[1] > .8, "reversed")],
}


def draw_instances(cond, n, rng):
    out = []
    while len(out) < n:
        x = rng.random(P)
        if cond(x):
            out.append(x)
    return np.array(out)


def rule_from_attributions(m, I, partner_main, tau, use_partner):
    """Map (first-order effect m of x1, pair interaction I with x2, partner first-order) to a mechanism."""
    tot = m + I
    if abs(m) <= tau and abs(tot) <= tau:
        return "inactive"
    if abs(I) <= max(tau, KAPPA * max(abs(m), abs(tot))):
        return "standalone"
    if abs(m) > tau and abs(tot) > tau and np.sign(m) != np.sign(tot):
        return "reversed"
    if abs(tot) > abs(m):
        return "amplified"
    if not use_partner:
        return "attenuated"
    return "redundant" if np.sign(partner_main) == np.sign(m) else "gated"


def run(scen, seed):
    import shapiq
    rng = np.random.default_rng(seed)
    f = FUNCS[scen]
    bg = pd.DataFrame(rng.random((2000, P)), columns=COLS)
    rows = []
    for level in ["DGP", "RF", "XGB", "ANN"]:
        model = fit(level, f, scen, seed, np.random.default_rng(seed + 100))
        tau = 0.1 * (np.std(model.predict(bg.iloc[:500])) + 1e-12)   # same relevance threshold as CBFI labels
        pred = lambda A: model.predict(pd.DataFrame(A, columns=COLS))
        ksii = shapiq.TabularExplainer(model=pred, data=bg.sample(100, random_state=seed).to_numpy(),
                                       index="k-SII", max_order=2, random_state=seed)
        te = None
        if level in ("RF", "XGB"):
            import shap
            te = shap.TreeExplainer(model)
        for sub, cond, gt in GROUPS[scen]:
            Xi = draw_instances(cond, N_PER_GROUP, rng)
            siv = te.shap_interaction_values(pd.DataFrame(Xi, columns=COLS)) if te is not None else None
            for k, x in enumerate(Xi):
                xs = pd.Series(x, index=COLS)
                r = V2.decompose_feature(model, xs, bg, "x1", n_samples=N_SAMPLES, rng=seed * 1000 + k, kappa=KAPPA)
                preds = {"CBFI": r["label"]}
                iv = ksii.explain(x, budget=2 ** P)
                m1, m2, I = iv[(0,)], iv[(1,)], iv[(0, 1)]
                preds["kSII (sign)"] = rule_from_attributions(m1, I, m2, tau, False)
                preds["kSII (+partner)"] = rule_from_attributions(m1, I, m2, tau, True)
                if siv is not None:
                    S = siv[k]
                    preds["TreeSHAP-IQ (sign)"] = rule_from_attributions(S[0, 0], 2 * S[0, 1], S[1, 1], tau, False)
                    preds["TreeSHAP-IQ (+partner)"] = rule_from_attributions(S[0, 0], 2 * S[0, 1], S[1, 1], tau, True)
                for meth, p in preds.items():
                    rows.append({"scenario": scen, "subgroup": sub, "level": level, "seed": seed, "instance": k,
                                 "method": meth, "truth": gt, "pred": p})
    print(f"done {scen} seed={seed}", flush=True)
    return rows


if __name__ == "__main__":
    out = Parallel(n_jobs=18)(delayed(run)(s, sd) for s in FUNCS for sd in range(3))
    save(pd.DataFrame([r for rows in out for r in rows]), "mechanism.csv")
