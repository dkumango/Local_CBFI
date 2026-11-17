"""
Is the case taxonomy new information, or can it be reconstructed from Shapley-type attributions?

Part A — synthetic mechanism benchmark (same scenarios, seeds and instances as exp_mechanism.py).
  Adds PARTNER-FREE Shapley baselines that use the Shapley analogue of every CBFI quantity:
    m    : first-order / main effect of x1               (analogue of A, standalone)
    tot  : m + all pair interactions involving x1        (analogue of B, in context)
    rest : attribution of all other features (incl. their pairs)  (analogue of C)
  rule "rest-sign" : attenuated -> redundant if sign(rest) == sign(m), else gated
  rule "rest-half" : attenuated -> redundant if rest*sign(m) >= |tot - m|/2 (exact CBFI rule), else gated
  for k-SII (all levels) and TreeSHAP interaction values (RF, XGB).
  Output: results/mechanism_v2.csv  (all methods of exp_mechanism.py + the new ones)

Part B — real data (RF; insurance, wine, california, bike; 60 test instances x all features).
  For every (instance, feature): CBFI label + confidence, and TreeSHAP quantities phi_i, phi_ii, rest_i.
  Output: results/richness_real.csv
  Analysis (in make_figures / notes): (i) same-attribution pairs with different labels,
  (ii) how well a classifier on Shapley quantities reconstructs the CBFI label.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_cbfi_v2 as V2
from common import save, split, fit_model, background
from exp_taxonomy import fit, COLS, P
from exp_mechanism import FUNCS, GROUPS, draw_instances, rule_from_attributions, KAPPA, N_PER_GROUP, N_SAMPLES


def rule_rest(m, tot, rest, tau, variant):
    """Partner-free Shapley analogue of the CBFI rule (m ~ A, tot ~ B, rest ~ C)."""
    I = tot - m
    if abs(m) <= tau and abs(tot) <= tau:
        return "inactive"
    if abs(I) <= max(tau, KAPPA * max(abs(m), abs(tot))):
        return "standalone"
    if abs(m) > tau and abs(tot) > tau and np.sign(m) != np.sign(tot):
        return "reversed"
    if abs(tot) > abs(m):
        return "amplified"
    if variant == "rest-sign":
        return "redundant" if np.sign(rest) == np.sign(m) else "gated"
    return "redundant" if rest * np.sign(m) >= abs(I) / 2 else "gated"


def run_mech(scen, seed):
    """Mirrors exp_mechanism.run (identical RNG call order -> identical models and instances)."""
    import shapiq
    rng = np.random.default_rng(seed)
    f = FUNCS[scen]
    bg = pd.DataFrame(rng.random((2000, P)), columns=COLS)
    rows = []
    for level in ["DGP", "RF", "XGB", "ANN"]:
        model = fit(level, f, scen, seed, np.random.default_rng(seed + 100))
        tau = 0.1 * (np.std(model.predict(bg.iloc[:500])) + 1e-12)
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
                # partner-free: all pairs with x1, and the attribution of all other features
                I_all = sum(iv[(0, j)] for j in range(1, P))
                rest = sum(iv[(j,)] for j in range(1, P)) + sum(iv[(j, l)] for j in range(1, P) for l in range(j + 1, P))
                for v in ("rest-sign", "rest-half"):
                    preds[f"kSII ({v})"] = rule_rest(m1, m1 + I_all, rest, tau, v)
                if siv is not None:
                    S = siv[k]
                    preds["TreeSHAP-IQ (sign)"] = rule_from_attributions(S[0, 0], 2 * S[0, 1], S[1, 1], tau, False)
                    preds["TreeSHAP-IQ (+partner)"] = rule_from_attributions(S[0, 0], 2 * S[0, 1], S[1, 1], tau, True)
                    phi = S.sum(axis=1)
                    for v in ("rest-sign", "rest-half"):
                        preds[f"TreeSHAP-IQ ({v})"] = rule_rest(S[0, 0], phi[0], phi.sum() - phi[0], tau, v)
                for meth, p in preds.items():
                    rows.append({"scenario": scen, "subgroup": sub, "level": level, "seed": seed, "instance": k,
                                 "method": meth, "truth": gt, "pred": p})
    print(f"done mech {scen} seed={seed}", flush=True)
    return rows


def run_real(ds, n_inst=60):
    import shap
    Xtr, Xte, ytr, yte, task = split(ds)
    model = fit_model("RF", task, Xtr, ytr)
    bg = background(Xtr)
    est, sc = model.named_steps["est"], model.named_steps["scale"]
    te = shap.TreeExplainer(est)
    Xi = Xte.iloc[:n_inst]
    siv = np.asarray(te.shap_interaction_values(sc.transform(Xi)))
    if task == "clf":
        cls = model.predict(Xi)
        ci = np.searchsorted(est.classes_, cls)
        if siv.ndim == 4 and siv.shape[0] == len(est.classes_):     # (K, n, p, p) layout
            siv = np.moveaxis(siv, 0, -1)
        siv = np.stack([siv[k, :, :, ci[k]] for k in range(len(Xi))])
        vbg = model.predict_proba(bg)
        sd = float(np.std(vbg[np.arange(len(bg)), np.searchsorted(est.classes_, model.predict(bg))]))
    else:
        sd = float(np.std(model.predict(bg)))

    def one(i):
        x = Xi.iloc[i]
        S = siv[i]
        phi = S.sum(axis=1)
        out = []
        for j, feat in enumerate(x.index):
            r = V2.decompose_feature(model, x, bg, feat, task=task, n_samples=200, rng=7000 + i, return_samples=True)
            out.append({"dataset": ds, "instance": i, "feature": feat, "sd": sd,
                        "label": r["label"], "label_conf": r["label_conf"],
                        "G1": r["G1"], "G4": r["G4"], "Imp": r["Imp"], "C": float(r["samples"].c.mean()),
                        "phi": phi[j], "phi_main": S[j, j], "phi_int": phi[j] - S[j, j], "rest": phi.sum() - phi[j],
                        **({"smoker": x["smoker"], "bmi": x["bmi"]} if ds == "insurance" else {})})
        return out
    parts = Parallel(n_jobs=20)(delayed(one)(i) for i in range(len(Xi)))
    print(f"done real {ds}", flush=True)
    return [r for p in parts for r in p]


if __name__ == "__main__":
    real = []
    for ds in ["insurance", "wine", "california", "bike"]:
        real += run_real(ds)
    save(pd.DataFrame(real), "richness_real.csv")
    out = Parallel(n_jobs=18)(delayed(run_mech)(s, sd) for s in FUNCS for sd in range(3))
    save(pd.DataFrame([r for rows in out for r in rows]), "mechanism_v2.csv")
