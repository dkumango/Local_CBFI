"""
Figures (PNG 300 dpi + PDF) and tables (CSV + one DOCX) for the revised manuscript.
Reads revision_exp/results/*.csv; writes to submit_PeerJ_CS_260603/revision_1/figures_v2 and tables_v2.
Palette: reference categorical order (blue, orange, aqua, yellow), light surface; markers / line styles
are added as secondary encoding so the figures survive grayscale printing.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "submit_PeerJ_CS_260603", "revision_1")
FIG, TAB = os.path.join(OUT, "figures_v2"), os.path.join(OUT, "tables_v2")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "violet": "#6250d6", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
METHOD_STYLE = {   # color, marker, linestyle
    "CBFI": (C["blue"], "o", "-"), "kSII": (C["orange"], "s", "--"), "TreeSHAP-IQ": (C["aqua"], "^", "-."),
    "SHAP": (C["orange"], "s", "--"), "LIME": (C["aqua"], "^", "-."), "random": (C["gray"], "", ":"),
}
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6, "axes.labelcolor": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.dpi": 300, "figure.dpi": 110,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure", name)


def r(name):
    return pd.read_csv(os.path.join(RES, name))


TABLES = []   # (title, DataFrame) for the DOCX


def table(df, name, title):
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    TABLES.append((name, title, df))
    print("table", name)


# ======================================================================================
# Fig 1 — framework schematic
# ======================================================================================
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_axis_off(); ax.set_xlim(0, 124); ax.set_ylim(0, 52)

    def box(x, y, w, h, text, fc="#f4f3f0", ec=C["ink2"], bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6,
                weight="bold" if bold else "normal", color=C["ink"], linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C["ink2"]))

    ax.text(1, 50, "(a) Hybrid states per background row z", fontsize=8.5, weight="bold")
    states = [("full", "f(x)"), ("minus i", "f(x$_{-i}$, z$_i$)"), ("alone i", "f(x$_i$, z$_{-i}$)"), ("none", "f(z)")]
    for k, (nm, ex) in enumerate(states):
        box(2 + k * 13, 36, 11.5, 9, f"{nm}\n{ex}")
    ax.text(1, 30, "(b) Per-sample effects", fontsize=8.5, weight="bold")
    box(2, 13.5, 50, 13, "", fc="#eef4fc", ec=C["blue"])
    eqs = [("b = f(x) − f(x$_{-i}$, z$_i$)", "in-context effect"), ("a = f(x$_i$, z$_{-i}$) − f(z)", "standalone effect"),
           ("c = f(x$_{-i}$, z$_i$) − f(z)", "effect of the rest"), ("d = b − a", "interaction effect")]
    for k, (eq, desc) in enumerate(eqs):
        yy = 24 - k * 2.9
        ax.text(4.5, yy, eq, fontsize=7.6, va="center", color=C["ink"])
        ax.text(31, yy, desc, fontsize=7.6, va="center", color=C["ink2"])
    arrow(27, 36, 27, 27.2)
    ax.text(1, 8.5, "G1 = mean(a),  G4 = mean(d)", fontsize=7.8, color=C["ink"])
    ax.text(1, 4.5, "Imp = mean(b) = G1 + G4", fontsize=7.8, color=C["ink"])
    ax.text(1, 0.5, "Label: rule on bootstrap means (τ, κ)", fontsize=7.6, color=C["ink2"])

    ax.text(62, 50, "(c) Case taxonomy (per instance and feature)", fontsize=8.5, weight="bold")
    rules = [("no detectable effect", "inactive", "#f4f3f0"),
             ("|d| small", "standalone (≈ CBFI G1)", "#f4f3f0"),
             ("sign(a) ≠ sign(b)", "reversed (suppression)", "#fdeee8"),
             ("|b| > |a|", "amplified (synergy, AND)", "#eef4fc"),
             ("|b| ≤ |a|, rest\nreproduces the effect", "redundant (OR; ≈ CBFI G3)", "#e8f7f1"),
             ("|b| ≤ |a|, otherwise", "gated (context switches i off)", "#fdf5e3")]
    for k, (cond, lab, fc) in enumerate(rules):
        y = 41 - k * 7.2
        ax.text(63, y + 2.3, cond, fontsize=7.2, color=C["ink2"], va="center", linespacing=1.1)
        box(89, y, 33, 4.6, lab, fc=fc)
        arrow(85.5, y + 2.3, 88.6, y + 2.3)
    arrow(53, 20, 60, 24)
    save(fig, "Fig01_framework")


# ======================================================================================
# RQ1 — synthetic benchmark
# ======================================================================================
def fig_synthetic():
    s = r("synthetic_summary.csv")
    s = s[s.method.isin(["CBFI", "kSII", "TreeSHAP-IQ"])]
    scen = ["pair", "suppressor", "two_pairs", "xor", "threeway", "conditional", "correlated"]
    levels = ["RF", "XGB", "ANN"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, lv in zip(axes, levels):
        d = s[s.level == lv]
        for j, m in enumerate(["CBFI", "kSII", "TreeSHAP-IQ"]):
            dm = d[d.method == m].set_index("scenario").reindex(scen)
            if dm["local_p@k"].isna().all():
                continue
            col, mk, _ = METHOD_STYLE[m]
            ax.scatter(dm["local_p@k"], np.arange(len(scen)) + (j - 1) * 0.22, color=col, marker=mk, s=22,
                       edgecolor="white", linewidth=0.6, label=m, zorder=3)
        ax.set_title(lv); ax.set_xlim(0.55, 1.02); ax.set_xlabel("local precision@k")
        ax.set_yticks(range(len(scen))); ax.set_yticklabels(scen); ax.invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, "Fig02_synthetic_precision")

    # accuracy vs runtime (one axis each; runtime on log x)
    agg = s[s.scenario != "additive"].groupby(["level", "method"])[["local_p@k", "sec_per_instance"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    lv_mark = {"DGP": "o", "RF": "s", "XGB": "^", "ANN": "D"}
    for _, row in agg.iterrows():
        col = METHOD_STYLE[row.method][0]
        ax.scatter(row.sec_per_instance, row["local_p@k"], color=col, marker=lv_mark[row.level], s=34,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(f"{row.level}", (row.sec_per_instance, row["local_p@k"]), xytext=(4, -3),
                    textcoords="offset points", fontsize=6.8, color=C["ink2"])
    ax.set_xscale("log"); ax.set_xlabel("seconds per instance (full 8×8 pair matrix, log scale)")
    ax.set_ylabel("local precision@k (7 scenarios)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=METHOD_STYLE[m][0], label=m) for m in ["CBFI", "kSII", "TreeSHAP-IQ"]],
              loc="lower right")
    save(fig, "Fig03_accuracy_vs_runtime")

    # Table: synthetic summary
    s2 = r("synthetic_summary.csv")
    loc = s2[(s2.method != "H-stat (global)") & (s2.scenario != "additive")].groupby(["level", "method"]).agg(
        local_AUROC=("local_auroc", "mean"), local_Pk=("local_p@k", "mean"),
        max_null_over_sd=("max_null_pair/sd(f)", "mean"), corr_analytic=("corr_with_analytic", "mean"),
        sign_agree=("sign_agree", "mean"), sec_per_instance=("sec_per_instance", "mean")).reset_index()
    add = s2[s2.scenario == "additive"].set_index(["level", "method"])["max_null_pair/sd(f)"].rename("additive_max_null_over_sd")
    loc = loc.join(add, on=["level", "method"])
    h = s2[s2.method == "H-stat (global)"].groupby("level")[["global_p@k"]].mean().rename(columns={"global_p@k": "H_global_Pk"})
    loc = loc.join(h, on="level")
    order = {"DGP": 0, "RF": 1, "XGB": 2, "ANN": 3}
    loc = loc.sort_values(["level", "method"], key=lambda c: c.map(order) if c.name == "level" else c)
    table(loc.round(3), "Table02_synthetic_summary",
          "Interaction recovery on synthetic functions (mean over 7 scenarios, 5 seeds, 20 instances).")
    st = r("synthetic_stability.csv").round(3)
    table(st, "Table03_explainer_stability",
          "Explainer stability under resampled background data (two_pairs, RF, 10 instances × 10 repeats).")


# ======================================================================================
# Mechanism identification + taxonomy robustness
# ======================================================================================
def fig_mechanism():
    m = r("mechanism_v2.csv")
    m["ok"] = m["truth"] == m["pred"]
    m = m[m.level != "DGP"]
    meths = [("CBFI", "CBFI (no partner info)", C["blue"], None),
             ("kSII (sign)", "k-SII, sign only", C["orange"], "////"),
             ("kSII (rest-sign)", "k-SII + rest attribution (no partner info)", C["violet"], "xx"),
             ("kSII (+partner)", "k-SII + partner sign", C["aqua"], "...")]
    types = ["amplified", "gated", "redundant", "reversed", "standalone"]
    acc = m.groupby(["truth", "method"]).ok.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    w = 0.2
    for j, (key, lab, col, hatch) in enumerate(meths):
        vals = acc.loc[types, key].values
        bars = ax.bar(np.arange(len(types)) + (j - 1.5) * (w + 0.01), vals, w, color=col, label=lab,
                      edgecolor="white", linewidth=0.8, hatch=hatch)
        for b, v in zip(bars, vals):
            if v < 0.05:
                ax.text(b.get_x() + b.get_width() / 2, 0.02, "0", ha="center", va="bottom", fontsize=7, color=C["ink2"])
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types)
    ax.set_ylabel("identification accuracy"); ax.set_ylim(0, 1.08)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    ax.grid(axis="x", visible=False)
    save(fig, "Fig04_mechanism_identification")

    order = ["CBFI", "kSII (sign)", "kSII (rest-half)", "kSII (rest-sign)", "kSII (+partner)",
             "TreeSHAP-IQ (sign)", "TreeSHAP-IQ (rest-half)", "TreeSHAP-IQ (rest-sign)", "TreeSHAP-IQ (+partner)"]
    lv = m.groupby(["method", "level"]).ok.mean().unstack()[["RF", "XGB", "ANN"]]
    by_type = m.groupby(["method", "truth"]).ok.mean().unstack()[types]
    t = lv.join(by_type).loc[order].reset_index().round(3)
    table(t, "Table04_mechanism_identification",
          "Accuracy of identifying the interaction mechanism of x1 (semantic ground truth; RF, XGB, ANN; 3 seeds). "
          "sign: first-order and pair value of x1; rest-sign / rest-half: the CBFI rule applied to Shapley quantities, "
          "with the attribution of all other features in place of C (no partner information); +partner: sign of the true partner's first-order value.")


def fig_richness():
    """Information beyond the attribution score (real data, RF)."""
    pairs = r("richness_summary_pairs.csv")
    rec = r("richness_summary_reconstruct.csv")
    ds_order = ["insurance", "wine", "california", "bike"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.45})
    ax = axs[0]
    pp = pairs[pairs.attr == "phi"].set_index("dataset").loc[ds_order]
    bars = ax.bar(range(4), pp.share_diff, 0.6, color=C["blue"], edgecolor="white")
    for b, (v, n) in zip(bars, zip(pp.share_diff, pp.pairs)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(ds_order)
    ax.set_ylim(0, 0.85); ax.set_ylabel("share with different labels")
    ax.set_title("A  Pairs with near-identical SHAP value", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axs[1]
    inputs = ["majority class", "TreeSHAP (phi_i)", "TreeSHAP (phi_i, main, int)", "TreeSHAP (phi_i, main, int, rest)", "CBFI means (A, B, C)"]
    labs = ["majority", "SHAP φᵢ", "+ main, interaction", "+ rest attribution", "CBFI (A, B, C)"]
    cols = [C["gray"], C["orange"], C["yellow"], C["violet"], C["blue"]]
    for k, subset in enumerate(["all labels", "redundant vs gated"]):
        s = rec[rec.subset == subset].set_index("inputs").loc[inputs]
        for j, (v, c) in enumerate(zip(s.accuracy, cols)):
            ax.bar(k * 6 + j, v, 0.85, color=c, edgecolor="white", label=labs[j] if k == 0 else None)
            ax.text(k * 6 + j, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([2, 8]); ax.set_xticklabels(["all labels\n(n=812)", "redundant vs gated\n(n=170)"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("CV accuracy")
    ax.set_title("B  Reconstructing the label from attributions", loc="left")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "Fig11_information_beyond_attribution")
    t = rec.copy()
    t["inputs"] = t.inputs.replace(dict(zip(inputs, labs)))
    table(t, "Table13_label_reconstruction",
          "Predicting the localized CBFI label from TreeSHAP quantities (gradient boosting, 5-fold CV grouped by instance; "
          "RF on insurance, wine, california and bike; 60 test instances; non-inactive labels with bootstrap confidence >= 0.8).")


def fig_taxonomy():
    d = r("taxonomy_synthetic.csv")
    d = d[d.scenario != "duplicate"]
    d["ok"] = d["gt"] == d["pred"]
    k = d[(d.method == "instance tau=0.1 k=0.1") & (d.level == "RF")]
    labs = ["inactive", "standalone", "amplified", "redundant", "gated", "reversed"]
    cm = pd.crosstab(k["gt"], k["pred"], normalize="index").reindex(index=labs, columns=labs).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = cm.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 0.55 else C["ink"])
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("label on fitted RF"); ax.set_ylabel("label on true function"); ax.grid(False)
    ax.set_title("(a) Row-normalised confusion, τ = 0.1, κ = 0.1")

    ax = axes[1]
    taus = [0.01, 0.05, 0.1, 0.2]
    for lv, (col, mk, ls) in zip(["RF", "XGB", "ANN"], [METHOD_STYLE["CBFI"], METHOD_STYLE["kSII"], METHOD_STYLE["TreeSHAP-IQ"]]):
        y = [d[(d.level == lv) & (d.method == f"instance tau={t} k=0.1")].ok.mean() for t in taus]
        ax.plot(taus, y, color=col, marker=mk, ls=ls, label=lv)
    base = d[d.method == "per-sample majority"].groupby("level").ok.mean().mean()
    ax.axhline(base, color=C["gray"], ls=":", lw=1.2)
    ax.text(0.012, base + 0.015, f"per-sample majority ({base:.2f})", fontsize=7, color=C["ink2"])
    ax.set_xscale("log"); ax.set_xticks(taus); ax.set_xticklabels([str(t) for t in taus])
    ax.set_xlabel("practical threshold τ (fraction of sd of f)"); ax.set_ylabel("label accuracy")
    ax.set_ylim(0.35, 1.0); ax.set_title("(b) Sensitivity to τ")
    ax.legend(loc="lower right")
    save(fig, "Fig05_taxonomy_robustness")

    rows = []
    for meth in ["per-sample majority"] + [f"instance tau={t} k=0.1" for t in taus] + \
                [f"instance tau=0.1 k={kk}" for kk in [0.0, 0.05, 0.2]]:
        dm = d[d.method == meth]
        from sklearn.metrics import f1_score
        row = {"method": meth}
        for lv in ["RF", "XGB", "ANN"]:
            g = dm[dm.level == lv]
            row[f"acc_{lv}"] = g.ok.mean()
            row[f"macroF1_{lv}"] = f1_score(g["gt"], g["pred"], average="macro")
        rows.append(row)
    table(pd.DataFrame(rows).round(3), "Table05_taxonomy_sensitivity",
          "Agreement of case labels on fitted models with labels on the true function (8 scenarios × 3 seeds × 40 instances × 6 features).")


# ======================================================================================
# RQ2 — stability and scalability
# ======================================================================================
def fig_stability_scalability():
    c = r("stability_convergence.csv")
    g = c.groupby(["model", "n"]).cv_G4_rel_Imp.median().unstack(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.fill_between(g.index, g.min(1), g.max(1), color=C["blue"], alpha=0.15, lw=0, label="range over 6 models")
    ax.plot(g.index, g.median(1), color=C["blue"], marker="o", label="median over models")
    ax.axhline(0.10, color=C["gray"], ls=":", lw=1.2); ax.text(12, 0.108, "10%", fontsize=7, color=C["ink2"])
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100, 200, 500]); ax.set_xticklabels(["10", "25", "50", "100", "200", "500"])
    ax.set_xlabel("background samples n"); ax.set_ylabel("sd(G4) / |Imp|"); ax.set_title("(a) Monte Carlo variability")
    ax.legend(fontsize=7, loc="upper right")

    sc = r("scalability.csv")
    a = sc.groupby(["p", "method"])[["recall@5", "sec_per_instance"]].mean().reset_index()
    styles = {"CBFI two-stage": (C["blue"], "o", "-"), "CBFI full": (C["blue"], "o", ":"),
              "TreeSHAP-IQ": (C["aqua"], "^", "-."), "kSII (budget 2000)": (C["orange"], "s", "--")}
    for ax, col, lab, title in [(axes[1], "recall@5", "recall of the 5 true pairs", "(b) Pair recall"),
                                (axes[2], "sec_per_instance", "seconds per instance (log)", "(c) Runtime")]:
        for m, (cc, mk, ls) in styles.items():
            dm = a[a.method == m]
            ax.plot(dm.p, dm[col], color=cc, marker=mk, ls=ls, label=m.replace(" (budget 2000)", ""),
                    mfc="white" if m == "CBFI full" else cc)
        ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100, 200]); ax.set_xticklabels(["10", "25", "50", "100", "200"])
        ax.set_xlabel("number of features p"); ax.set_ylabel(lab); ax.set_title(title)
    axes[1].set_ylim(0, 1); axes[2].set_yscale("log")
    axes[2].legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "Fig06_stability_scalability")

    conv = c.groupby("n")[["cv_G4_rel_Imp", "cv_Imp"]].median().reset_index()
    conv["median_abs_dev_from_n1000_rel_Imp"] = c.assign(dev=(c.G4_mean - c.G4_ref).abs() / (c.Imp_ref.abs() + 1e-12)).groupby("n").dev.median().values
    table(conv.round(3), "Table06_convergence",
          "Monte Carlo variability of localized CBFI vs number of background samples (6 datasets × 6 models × 5 instances × 30 repeats; medians).")
    scal = a.pivot(index="p", columns="method", values=["recall@5", "sec_per_instance"]).round(3)
    scal.columns = [f"{x}|{y}" for x, y in scal.columns]
    scr = sc[sc.method == "CBFI two-stage"].groupby("p").screen_recall.mean().rename("CBFI screen recall (interacting features in top-12)")
    table(scal.join(scr).reset_index(), "Table08_scalability",
          "Scalability: pair recall and runtime vs number of features (5 true pairs; RF and XGB; 3 seeds × 5 instances).")
    smp = r("stability_sampler.csv")
    smp = smp[smp.sampler != "marginal"].groupby(["dataset", "sampler"])[["kendall_Imp_vs_marginal", "top3_Imp_overlap", "G4_share"]].mean().round(3).reset_index()
    table(smp, "Table09_sampler_sensitivity",
          "Sensitivity to the sampling strategy: agreement of conditional (KNN) with marginal sampling (RF, 20 instances).")


# ======================================================================================
# RQ4 — case studies
# ======================================================================================
def fig_cases():
    s = r("cases_insurance_smoker.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for sm, col, mk, lab in [(1, C["orange"], "s", "smokers"), (0, C["blue"], "o", "non-smokers")]:
        d = s[s.smoker == sm]
        ax.scatter(d.bmi, d.G4, color=col, marker=mk, s=26, edgecolor="white", linewidth=0.7, label=lab, zorder=3)
    ax.axvline(30, color=C["gray"], ls=":", lw=1.2); ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.text(30.4, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0, "BMI 30", fontsize=7, color=C["ink2"])
    ax.set_xlabel("BMI"); ax.set_ylabel("interaction effect G4 of smoker (USD)")
    ax.set_title("(a) Sign of the smoker interaction flips at BMI 30"); ax.legend(loc="lower left", fontsize=7)
    ax = axes[1]
    for sm, col, mk, lab in [(1, C["orange"], "s", "smokers"), (0, C["blue"], "o", "non-smokers")]:
        d = s[s.smoker == sm]
        ax.scatter(d.counterfactual_flip, d.Imp, color=col, marker=mk, s=26, edgecolor="white", linewidth=0.7, label=lab, zorder=3)
    ax.axhline(0, color=C["ink2"], lw=0.6); ax.axvline(0, color=C["ink2"], lw=0.6)
    rr = np.corrcoef(s.Imp, s.counterfactual_flip)[0, 1]
    ax.text(0.04, 0.92, f"Pearson r = {rr:.3f}", transform=ax.transAxes, fontsize=7.5, color=C["ink"])
    ax.set_xlabel("f(x) − f(x, smoker flipped)  (USD)"); ax.set_ylabel("in-context effect Imp of smoker (USD)")
    ax.set_title("(b) Agreement with a counterfactual check")
    fig.tight_layout()
    save(fig, "Fig08_insurance_smoker")

    # decomposition bars (standalone vs interaction magnitude)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, ds, unit in [(axes[0], "insurance", "USD"), (axes[1], "wine", "probability")]:
        d = r(f"cases_{ds}_decomp.csv")
        g = d.assign(aG1=d.G1.abs(), aG4=d.G4.abs(), aImp=d.Imp.abs()).groupby("feature")[["aG1", "aG4", "aImp"]].mean()
        g = g.sort_values("aImp").tail(8)
        y = np.arange(len(g))
        ax.barh(y + 0.2, g.aG1, 0.38, color=C["blue"], label="|standalone G1|", edgecolor="white", linewidth=0.6)
        ax.barh(y - 0.2, g.aG4, 0.38, color=C["orange"], label="|interaction G4|", edgecolor="white", linewidth=0.6, hatch="////")
        ax.set_yticks(y); ax.set_yticklabels(g.index); ax.set_xlabel(f"mean over 50 instances ({unit})")
        ax.set_title(f"({'a' if ds == 'insurance' else 'b'}) {ds.capitalize()} (RF)"); ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "Fig07_case_decomposition")

    # pairwise interaction heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, ds in [(axes[0], "insurance"), (axes[1], "wine")]:
        p = r(f"cases_{ds}_pairwise.csv")
        M = p.pivot(index="fi", columns="fj", values="mean_abs_G4ij")
        top = (M.sum(1)).sort_values(ascending=False).index[:7]
        M = M.loc[top, top]
        im = ax.imshow(M.values, cmap=cmap)
        for i in range(len(top)):
            for j in range(len(top)):
                if i != j:
                    v = M.values[i, j]; rel = v / np.nanmax(M.values)
                    ax.text(j, i, f"{v:.0f}" if ds == "insurance" else f"{v:.3f}", ha="center", va="center",
                            fontsize=6, color="white" if rel > 0.55 else C["ink"])
        ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7); ax.grid(False)
        ax.set_title(f"({'a' if ds == 'insurance' else 'b'}) {ds.capitalize()}: mean |G4$_{{ij}}$| (30 instances)")
    fig.tight_layout()
    save(fig, "Fig09_pairwise_interaction")

    # fidelity / deletion curves
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, ds, lab in [(axes[0], "wine", "drop in prob. of predicted class"), (axes[1], "insurance", "|shift| of prediction (USD)")]:
        f = r(f"cases_{ds}_fidelity.csv")
        cur = f.groupby(["method", "k"]).value.mean().unstack(0)
        for m, name in [("Imp", "CBFI"), ("SHAP", "SHAP"), ("LIME", "LIME"), ("random", "random")]:
            col, mk, ls = METHOD_STYLE[name]
            ax.plot(cur.index, cur[m], color=col, marker=mk or None, ls=ls, label=name, markersize=3.5)
        ax.set_xlabel("number of top-ranked features replaced (k)"); ax.set_ylabel(lab)
        ax.set_title(f"({'a' if ds == 'wine' else 'b'}) {ds.capitalize()} (50 instances)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "Fig10_fidelity")

    # tables: agreement + fidelity AUC
    rows = []
    for ds in ["wine", "insurance"]:
        a = r(f"cases_{ds}_agreement.csv").groupby("pair")[["spearman", "kendall", "top3_jaccard", "rbo"]].mean()
        a.insert(0, "dataset", ds); rows.append(a.reset_index())
    table(pd.concat(rows).round(3), "Table10_rank_agreement",
          "Rank agreement between explanations (|values|; mean over 50 instances; RF).")
    rows = []
    for ds in ["wine", "insurance"]:
        f = r(f"cases_{ds}_fidelity.csv")
        auc = f.groupby("method").value.mean()
        rows.append({"dataset": ds, **auc.to_dict()})
    table(pd.DataFrame(rows).round(4), "Table11_fidelity_auc",
          "Deletion fidelity: mean output change over k (higher = the ranking removes more of the decision; 50 instances).")


def tab_multidomain():
    m = r("multidomain_summary.csv")
    cols = ["dataset", "model", "p", "n_train", "score", "G4_share", "interaction_activation",
            "sec_cbfi", "sec_shap_perm", "sec_lime", "sec_treeshap", "Imp~SHAP_kendall", "Imp~SHAP_rbo"]
    order_ds = ["adult", "bank", "diabetes", "california", "bike", "ames"]
    order_m = ["RF", "XGB", "CAT", "ANN", "SVM", "TabNet"]
    m = m[cols].copy()
    m["dataset"] = pd.Categorical(m.dataset, order_ds); m["model"] = pd.Categorical(m.model, order_m)
    table(m.sort_values(["dataset", "model"]).round(3), "Table12_multidomain",
          "Multi-domain results (10 test instances per dataset-model pair; n = 100).")
    # runtime figure-free summary per model
    rt = r("multidomain_summary.csv").groupby("model")[["sec_cbfi", "sec_shap_perm", "sec_lime", "sec_treeshap"]].mean().round(3)
    table(rt.reset_index(), "Table07_runtime", "Mean runtime per explained instance (s), averaged over six datasets.")


def tab_datasets():
    sys.path.insert(0, HERE)
    from common import DATASETS
    info = {"wine": ("Wine Quality (red)", "UCI"), "insurance": ("Medical Insurance", "Lantz 2013"),
            "adult": ("Adult Census Income", "UCI"), "bank": ("Bank Marketing (bank.csv)", "UCI"),
            "diabetes": ("Diabetes Health Indicators (50/50)", "Kaggle / BRFSS 2015"),
            "california": ("California Housing", "scikit-learn"), "bike": ("Bike Sharing (daily)", "UCI"),
            "ames": ("Ames Housing", "De Cock 2011")}
    rows = []
    for k, fn in DATASETS.items():
        try:
            X, y, task = fn()
        except Exception as e:
            print("skip", k, e); continue
        rows.append({"key": k, "dataset": info[k][0], "task": "classification" if task == "clf" else "regression",
                     "instances": len(X), "features (after encoding)": X.shape[1],
                     "target": f"{y.nunique()} classes" if task == "clf" else "continuous", "source": info[k][1]})
    table(pd.DataFrame(rows), "Table01_datasets", "Datasets used in this study.")


def write_docx():
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(8)
    for name, title, df in sorted(TABLES):
        doc.add_paragraph().add_run(f"{name.replace('_', ' ', 1)}. {title}").bold = True
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"
        for j, c in enumerate(df.columns):
            t.rows[0].cells[j].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
        doc.add_paragraph()
    p = os.path.join(TAB, "tables_v2_all.docx")
    doc.save(p); print("docx", p)


if __name__ == "__main__":
    fig_framework(); fig_synthetic(); fig_mechanism(); fig_taxonomy()
    fig_stability_scalability(); fig_cases(); tab_multidomain(); tab_datasets(); fig_richness()
    write_docx()
