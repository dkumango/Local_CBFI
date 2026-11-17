# Localized CBFI — how does a feature act in an individual prediction?

Code and results for the revised PeerJ Computer Science manuscript
*"How do features act in an individual prediction? Separating standalone and interaction effects with a case-based taxonomy"*.

Attribution methods such as SHAP and LIME give each feature one number for a prediction, for example "smoking raises the predicted charge by 24,000 USD".
Localized CBFI adds two things to that number:

1. **a decomposition** of the feature's importance into a *standalone* part (what the feature does on its own) and an *interaction* part (what it does only because of the other feature values of this instance);
2. **a label** that states *how* the feature acts in this prediction: `standalone`, `amplified`, `redundant`, `gated`, `reversed` or `inactive`.

The method is model-agnostic: it only needs `model.predict` (regression) or `model.predict_proba` (classification). True labels are not used.

---

## Basic idea

### Four hybrid states
Let `x` be the instance to explain, `i` a feature, and `z` a row drawn from background data (by default, rows from the training data).
For each background row the model is evaluated in four states:

| state | input | meaning |
|---|---|---|
| full | `x` | all features from the instance |
| minus i | `x` with feature `i` taken from `z` | feature `i` removed, the rest of `x` kept |
| alone i | `z` with feature `i` taken from `x` | only feature `i` kept |
| none | `z` | nothing from the instance |

From these four predictions, per background row:

```
b = f(full)    - f(minus i)   in-context effect of i      (effect of i with the rest of x present)
a = f(alone i) - f(none)      standalone effect of i      (effect of i on its own)
c = f(minus i) - f(none)      effect of the remaining features of x without i
d = b - a                     interaction effect of i     (= 0 for every row if f is additive in i)
```

Averaging over `n` background rows gives

```
G1  = mean(a)               standalone effect
G4  = mean(d)               interaction effect
Imp = mean(b) = G1 + G4     in-context importance (exact decomposition)
```

All three are signed and in output units (e.g. USD, or probability of the predicted class).

### Case taxonomy
The label compares the standalone effect `A`, the in-context effect `B`, the effect of the remaining features `C` and the interaction `D` (means of `a`, `b`, `c`, `d`):

| label | rule (applied in this order) | meaning | original CBFI group |
|---|---|---|---|
| `inactive` | \|A\| and \|B\| below the relevance threshold τ | no relevant effect | (G2) |
| `standalone` | \|D\| small relative to \|A\|, \|B\| | acts the same alone and in context | G1 |
| `reversed` | sign(A) ≠ sign(B) | the context reverses the direction of the effect | — |
| `amplified` | \|B\| > \|A\| | the context strengthens the effect (synergy, AND) | G4 |
| `redundant` | C·sign(A) ≥ \|D\|/2 | the other features already provide the effect (substitution, OR) | G3 |
| `gated` | otherwise | the context switches the effect off (e.g. unmet AND condition) | — |

Defaults: τ = 0.1 × sd of the model output over the background data, relative threshold κ = 0.1.
On fitted models, the label is computed from 300 bootstrap replicates of the background rows; `label_conf` is the share of replicates that agree.

**Why this matters.** Two features can have the same SHAP value but act in different ways. In our experiments about half of feature pairs with nearly identical SHAP values received different labels. A pairwise interaction value alone cannot tell substitution (OR) from gating (AND with an unmet condition), because both give the same negative value. The effect of the remaining features (`c`) separates them.

### Pairwise interaction
To find *which* feature is the partner, localized CBFI also computes, with all other features kept at `x`,

```
G4_ij = mean[ f(x) - f(x without i) - f(x without j) + f(x without i, j) ]
```

which is zero when the model has no term coupling `i` and `j`. For many features, rank features by mean |d| first and compute `G4_ij` only among the top candidates (two-stage screening).

---

## Installation

The library is a single file, `local_cbfi_v2.py`. It needs only `numpy`, `pandas` and `scikit-learn`.

```bash
git clone https://github.com/dkumango/Local_CBFI.git
cd Local_CBFI
pip install numpy pandas scikit-learn
```

The experiment scripts need more packages (see [Environment](#environment)).

## Quick start

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import local_cbfi_v2 as V2

df = pd.read_csv("dataset/Medical_Cost.csv")
df["sex"] = (df.sex == "male").astype(int)
df["smoker"] = (df.smoker == "yes").astype(int)
df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)
X, y = df.drop(columns="charges"), df["charges"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_tr, y_tr)

background = X_tr.sample(1000, random_state=0)
x = X_te[(X_te.smoker == 1) & (X_te.bmi >= 30)].iloc[0]          # an obese smoker

# all features of one instance
res = V2.decompose_instance(model, x, background, task="regression", n_samples=200, rng=0)
print(res[["G1", "G4", "Imp", "label", "label_conf"]].head(4))

# which feature does smoking interact with?
print(V2.interaction_matrix(model, x, background, features=["age", "bmi", "children", "smoker"],
                            task="regression", n_samples=200, rng=0).round(0))
```

For classification, pass `task="clf"`; the explained output is the probability of the class predicted for `x`.

### Test script
`examples/quickstart.py` runs two checks (well under a minute, no GPU):

```bash
python examples/quickstart.py
```

**Part 1 — known functions.** `x1` is explained in functions whose logic is known, so the correct label is known in advance:

```
     function / instance  G1 (standalone)  G4 (interaction)  Imp = G1 + G4      label  confidence   expected
  additive  x1 + x2 + x3            0.409            -0.000          0.409 standalone       1.000 standalone
AND, both conditions met            0.265             0.252          0.517  amplified       1.000  amplified
 AND, x2 condition unmet            0.265            -0.265          0.000      gated       1.000      gated
  OR, both inputs active            0.252            -0.252          0.000  redundant       1.000  redundant
 2 x1 - 3 x1 x2, x2 high            0.201            -0.548         -0.348   reversed       1.000   reversed

All labels as expected: True
```

The additive function has no interaction (`G4 = 0`). The OR and unmet-AND cases have the same interaction value (−0.25), but different labels: in OR, `x2` already gives the effect (redundant); in AND, the unmet condition on `x2` switches `x1` off (gated).

**Part 2 — random forest on medical insurance.** For an obese smoker, smoking has a large standalone effect (about 17,600 USD) that BMI amplifies (`G4` ≈ +6,600 USD, label `amplified`). The pairwise matrix identifies BMI as the partner (`G4_ij` ≈ 6,100 USD). For a non-obese smoker, the same standalone effect is reduced by the context (`G4` ≈ −8,400 USD). Exact values can vary slightly with the scikit-learn version.

---

## Library reference (`local_cbfi_v2.py`)

| function | returns |
|---|---|
| `decompose_feature(model, x, background, feature, task="regression"\|"clf")` | `G1`, `G4`, `Imp`, mean \|d\| (`G4_abs`), `label`, `label_conf`, bootstrap interval of G4 (`G4_lo`, `G4_hi`), descriptive per-sample case shares (`share_*`); per-sample `a, b, c, d` with `return_samples=True` |
| `decompose_instance(model, x, background, task)` | one row per feature, sorted by \|Imp\| |
| `pairwise_interaction(model, x, background, fi, fj)` | `G4_ij`, mean \|d_ij\| |
| `interaction_matrix(model, x, background, features)` | symmetric matrix of `G4_ij` |
| `classify_effect(a, b, c, tau, kappa)` | instance-level label from per-sample effects |
| `draw_background(..., sampler="marginal"\|"conditional")` | background rows; `conditional` draws from the nearest neighbours of `x` |

Main arguments: `n_samples` (background rows per feature, default 100), `sampler`, `tau_label` (relevance threshold as a fraction of the output sd, default 0.1), `kappa` (default 0.1), `rng`.

Cost: about `3·n·p` model evaluations to explain all `p` features of one instance, batched into a few `predict` calls.

### Notes
- Localized CBFI is **not** more accurate than dedicated interaction indices for finding interacting pairs (k-SII, TreeSHAP interaction values), and it is slower than SHAP. Its purpose is the decomposition and the label.
- The label depends on the relevance threshold τ; report it with the explanation.
- Marginal background sampling describes how the *model* responds and can create unrealistic inputs for strongly correlated features. Try `sampler="conditional"` and report both when features are strongly correlated.
- The explanations describe the fitted model, not causal effects in the data.

---

## Reproducing the paper (`revision_exp/`)

| script | manuscript |
|---|---|
| `exp_synthetic.py` | RQ1: recovery of known interactions vs k-SII, TreeSHAP-IQ, H-statistic (Fig. 2–3, Tables 2–3) |
| `exp_mechanism.py`, `exp_taxonomy.py` | RQ2: mechanism identification and label robustness (Fig. 4–5, Tables 4–5) |
| `exp_richness.py`, `analyze_richness.py` | Partner-free Shapley baselines for the taxonomy (Fig. 4, Table 4) and information beyond the attribution score (Fig. 11, Table 13) |
| `exp_stability.py`, `exp_scalability.py` | RQ3/RQ4: Monte Carlo variability, scalability, sampling sensitivity (Fig. 6, Tables 6, 8, 9) |
| `exp_cases.py`, `exp_multidomain.py` | RQ5: case studies and multi-domain results (Fig. 7–10, Tables 7, 10–12) |
| `exp_redundancy.py` | supplementary check of the redundant / gated / reversed split |
| `make_figures.py` | regenerates all figures and tables from `results/` |

All result CSVs used in the paper are in `revision_exp/results/`.
Models are scikit-learn pipelines (StandardScaler → estimator) fitted on the training split only.

## Data
Wine quality, medical insurance and bike sharing are included in `dataset/`. Download the others into `dataset/`:
Adult (`adult.data.csv`, UCI), Bank Marketing (`bank.csv`, UCI), Diabetes Health Indicators
(`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`, Kaggle), Ames Housing (`AmesHousing.csv`).
California Housing is fetched through scikit-learn.

## Environment
Python 3.10; see `requirements_server.txt` (scikit-learn 1.7.1, shap 0.49.1, shapiq 1.4.1, lime 0.2.0.1,
xgboost 3.0.5, catboost 1.2.10, pytorch-tabnet 4.1.0). The experiments were run on 24 CPU cores with an RTX 3090.

## Reference
The original (global) CBFI: Oh S. 2022. Predictive case-based feature importance and interaction. *Information Sciences* 593:155–176.
