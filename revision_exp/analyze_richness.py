"""Analysis of exp_richness.py outputs -> printed tables + results/richness_summary_*.csv"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pd.set_option("display.width", 200)

# ---------- Part A: mechanism identification with partner-free Shapley baselines ----------
m = pd.read_csv(os.path.join(R, "mechanism_v2.csv"))
old = pd.read_csv(os.path.join(R, "mechanism.csv"))
k = ["scenario", "subgroup", "level", "seed", "instance", "method"]
mm = m.merge(old, on=k, suffixes=("", "_old"))
print("A0. identical predictions to mechanism.csv for shared methods:", round((mm.pred == mm.pred_old).mean(), 4), len(mm))
m["ok"] = m.pred == m.truth
acc = m.groupby(["method", "level"]).ok.mean().unstack()[["DGP", "RF", "XGB", "ANN"]].round(2)
print("\nA1. accuracy by method x level\n", acc)
by = m[m.level != "DGP"].groupby(["method", "truth"]).ok.mean().unstack().round(2)
print("\nA2. accuracy by true mechanism (fitted models)\n", by)
acc.to_csv(os.path.join(R, "richness_summary_mech_level.csv"))
by.to_csv(os.path.join(R, "richness_summary_mech_truth.csv"))

# ---------- Part B: real data ----------
d = pd.read_csv(os.path.join(R, "richness_real.csv"))
for c in ["phi", "phi_main", "phi_int", "rest", "G1", "G4", "Imp", "C"]:
    d[c + "_n"] = d[c] / d.sd
print("\nB0. label distribution\n", pd.crosstab(d.dataset, d.label))

rel = d[(d.label != "inactive") & (d.label_conf >= 0.8)].copy()
print("\nB1. relevant confident entries:", len(rel), "of", len(d))


def same_attr_pairs(df, col, rtol=0.10):
    rows = []
    for ds, g in df.groupby("dataset"):
        v, lab = g[col].to_numpy(), g.label.to_numpy()
        tot = diff = 0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if np.sign(v[i]) == np.sign(v[j]) and abs(v[i] - v[j]) <= rtol * max(abs(v[i]), abs(v[j])):
                    tot += 1
                    diff += lab[i] != lab[j]
        rows.append({"dataset": ds, "attr": col, "pairs": tot, "diff_label": diff, "share_diff": diff / max(tot, 1)})
    return pd.DataFrame(rows)


sp = pd.concat([same_attr_pairs(rel, "phi"), same_attr_pairs(rel, "Imp")])
print("\nB2. pairs with near-identical attribution (same sign, within 10%) that receive different labels\n", sp)
sp.to_csv(os.path.join(R, "richness_summary_pairs.csv"), index=False)

# reconstructibility: predict the CBFI label from Shapley quantities (grouped CV over instances)
rel["grp"] = rel.dataset + "_" + rel.instance.astype(str)
feats = {"TreeSHAP (phi_i)": ["phi_n"],
         "TreeSHAP (phi_i, main, int)": ["phi_n", "phi_main_n", "phi_int_n"],
         "TreeSHAP (phi_i, main, int, rest)": ["phi_n", "phi_main_n", "phi_int_n", "rest_n"],
         "CBFI means (A, B, C)": ["G1_n", "Imp_n", "C_n"]}
out = []
for subset, df in [("all labels", rel), ("redundant vs gated", rel[rel.label.isin(["redundant", "gated"])])]:
    y = df.label.to_numpy()
    for name, cols in feats.items():
        X = df[cols].to_numpy()
        pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(5).split(X, y, df.grp):
            pred[te] = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).predict(X[te])
        out.append({"subset": subset, "n": len(y), "inputs": name,
                    "accuracy": accuracy_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred)})
    maj = pd.Series(y).value_counts().iloc[0] / len(y)
    out.append({"subset": subset, "n": len(y), "inputs": "majority class", "accuracy": maj,
                "balanced_acc": 1 / df.label.nunique()})
rec = pd.DataFrame(out).round(3)
print("\nB3. reconstructing the CBFI label from attributions (5-fold CV grouped by instance)\n", rec)
rec.to_csv(os.path.join(R, "richness_summary_reconstruct.csv"), index=False)

# example: same phi, different labels (prefer insurance)
ex = []
for ds, g in rel.groupby("dataset"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.loc[i], g.loc[j]
            if a.label != b.label and np.sign(a.phi) == np.sign(b.phi) and abs(a.phi - b.phi) <= .05 * max(abs(a.phi), abs(b.phi)) \
                    and min(a.label_conf, b.label_conf) >= .95 and abs(a.phi_n) > .3:
                ex.append({"dataset": ds, "pair": f"{a.feature}@{a.instance} / {b.feature}@{b.instance}",
                           "phi": f"{a.phi:.3g} / {b.phi:.3g}", "labels": f"{a.label} / {b.label}",
                           "G1": f"{a.G1:.3g} / {b.G1:.3g}", "G4": f"{a.G4:.3g} / {b.G4:.3g}", "C": f"{a.C:.3g} / {b.C:.3g}",
                           "absphi_n": abs(a.phi_n)})
ex = pd.DataFrame(ex)
if len(ex):
    ex = ex.sort_values("absphi_n", ascending=False)
    print("\nB4. examples\n", ex.groupby("dataset").head(3).to_string())
    ex.to_csv(os.path.join(R, "richness_examples.csv"), index=False)
