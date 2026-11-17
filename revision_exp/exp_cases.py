"""
Case studies (Wine classification, Insurance regression; RandomForest) with localized CBFI v2.

Outputs (results/):
  cases_{ds}_decomp.csv        per instance x feature: G1, G4, Imp, case shares, SHAP, LIME
  cases_{ds}_pairwise.csv      mean |G4_ij| over instances + representative-instance G4_ij  (Fig. 7)
  cases_{ds}_agreement.csv     rank agreement CBFI vs SHAP vs LIME per instance                (E6)
  cases_{ds}_fidelity.csv      deletion curves (mean over instances) per ranking               (Figs. 5-6)
  cases_insurance_smoker.csv   smoker: G1/G4/Imp vs counterfactual flip, by BMI               (E5)
  cases_wine_alcohol_ice.csv   ICE of alcohol per instance, to check the 'all interaction' reading (R3)
  cases_meta.csv               accuracy / R2, representative indices
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import split, fit_model, score, background, save, SEED
import local_cbfi_v2 as V2
import xai_baselines as XB

N_INST, N_SAMPLES, N_PAIR_INST = 50, 200, 30
REPRESENTATIVE = {"wine": 8, "insurance": 6}       # indices used in the first submission


def explain_one(model, task, x, bg, Xtr, k):
    d = V2.decompose_instance(model, x, bg, task=task, n_samples=N_SAMPLES, rng=k)
    d = d.loc[x.index]
    d["SHAP"] = XB.shap_permutation(model, task, x, bg, seed=k)
    d["LIME"] = XB.lime_weights(model, task, x, Xtr, seed=k)
    d["instance"] = k
    return d.reset_index(names="feature")


def deletion_curve(model, task, x, bg, order, n_rep=50, rng=0):
    """Replace the top-k features (by `order`) with background values; mean output change vs. k."""
    v = V2.make_value_fn(model, x, task)
    v0 = float(v(pd.DataFrame([x]))[0])
    z = V2.draw_background(x, bg, None, n_rep, "marginal", rng=rng)
    out = [0.0]
    for k in range(1, len(order) + 1):
        h = z.copy()
        keep = [c for c in x.index if c not in order[:k]]
        for c in keep:
            h[c] = x[c]
        vk = v(h).mean()
        out.append(v0 - vk if task == "clf" else abs(v0 - vk))
    return np.array(out)       # clf: drop in prob of predicted class; reg: |shift| of prediction


def run_dataset(ds):
    Xtr, Xte, ytr, yte, task = split(ds)
    model = fit_model("RF", task, Xtr, ytr)
    acc = score(model, task, Xte, yte)
    bg = background(Xtr)
    rep = REPRESENTATIVE[ds]
    idx = [rep] + [i for i in range(len(Xte)) if i != rep][:N_INST - 1]

    parts = Parallel(n_jobs=20)(delayed(explain_one)(model, task, Xte.iloc[i], bg, Xtr, i) for i in idx)
    dec = pd.concat(parts, ignore_index=True)
    save(dec, f"cases_{ds}_decomp.csv")

    # rank agreement (E6)
    rows = []
    for i, g in dec.groupby("instance"):
        for a, b in [("Imp", "SHAP"), ("Imp", "LIME"), ("SHAP", "LIME"), ("G1", "SHAP"), ("G4", "SHAP")]:
            rows.append({"instance": i, "pair": f"{a} vs {b}", **XB.agreement(g[a], g[b])})
    save(pd.DataFrame(rows), f"cases_{ds}_agreement.csv")

    # fidelity / deletion (Figs 5-6), averaged over instances; random ordering as a floor
    rng = np.random.default_rng(SEED)
    curves = []
    for i, g in dec.groupby("instance"):
        x = Xte.iloc[i]
        for meth in ["Imp", "G1", "SHAP", "LIME", "random"]:
            order = list(rng.permutation(g.feature)) if meth == "random" else list(g.sort_values(meth, key=np.abs, ascending=False).feature)
            c = deletion_curve(model, task, x, bg, order, rng=i)
            curves += [{"instance": i, "method": meth, "k": k, "value": val} for k, val in enumerate(c)]
    save(pd.DataFrame(curves), f"cases_{ds}_fidelity.csv")

    # pairwise interaction (Fig. 7)
    def pm(i):
        return V2.interaction_matrix(model, Xte.iloc[i], bg, task=task, n_samples=N_SAMPLES, rng=i)
    mats = Parallel(n_jobs=20)(delayed(pm)(i) for i in idx[:N_PAIR_INST])
    mean_abs = sum(m.abs() for m in mats) / len(mats)
    pw = mean_abs.stack().rename("mean_abs_G4ij").to_frame()
    pw["rep_G4ij"] = mats[0].stack()
    save(pw.reset_index(names=["fi", "fj"]), f"cases_{ds}_pairwise.csv")

    if ds == "insurance":       # E5: counterfactual check for smoker
        rows = []
        sub = dec[dec.feature == "smoker"].set_index("instance")
        for i in idx:
            x = Xte.iloc[i]
            flip = x.copy(); flip["smoker"] = 1 - x["smoker"]
            cf = model.predict(pd.DataFrame([x]))[0] - model.predict(pd.DataFrame([flip]))[0]
            r = sub.loc[i]
            rows.append({"instance": i, "smoker": x["smoker"], "age": x["age"], "bmi": x["bmi"],
                         "G1": r.G1, "G4": r.G4, "Imp": r.Imp, "share_amplified": r.share_amplified,
                         "share_attenuated": r.share_attenuated, "counterfactual_flip": cf, "SHAP": r.SHAP})
        save(pd.DataFrame(rows), "cases_insurance_smoker.csv")

    if ds == "wine":            # R3: ICE of alcohol, to check the interpretation of alcohol's components
        grid = np.quantile(Xtr["alcohol"], np.linspace(0.05, 0.95, 15))
        rows = []
        for i in idx[:N_PAIR_INST]:
            x = Xte.iloc[i]
            v = V2.make_value_fn(model, x, task)
            H = pd.DataFrame([x] * len(grid)); H["alcohol"] = grid
            for gv, pv in zip(grid, v(H)):
                rows.append({"instance": i, "alcohol": gv, "prob_pred_class": pv, "x_alcohol": x["alcohol"]})
        save(pd.DataFrame(rows), "cases_wine_alcohol_ice.csv")

    return {"dataset": ds, "task": task, "score": acc, "n_train": len(Xtr), "n_test": len(Xte),
            "representative_index": rep, "n_instances": len(idx)}


if __name__ == "__main__":
    meta = [run_dataset(ds) for ds in ["insurance", "wine"]]
    save(pd.DataFrame(meta), "cases_meta.csv")
