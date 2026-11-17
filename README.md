# Localized CBFI — standalone and interaction effects with a case-based taxonomy

Code and results for the revised PeerJ Computer Science manuscript
"How do features act in an individual prediction? Separating standalone and interaction effects with a case-based taxonomy".

## Library
`local_cbfi_v2.py` — model-agnostic implementation.

| function | returns |
|---|---|
| `decompose_feature(model, x, background, feature, task="regression"/"clf")` | standalone effect `G1`, interaction effect `G4`, in-context importance `Imp = G1 + G4`, mean `|d|` (`G4_abs`), case label (`label`, `label_conf`), per-sample case shares |
| `decompose_instance(model, x, background, task)` | the above for every feature of `x` |
| `pairwise_interaction(...)`, `interaction_matrix(...)` | pairwise interaction `G4_ij` (Eq. 8) |
| `classify_effect(a, b, c, tau, kappa)` | instance-level label from per-sample effects |

Case labels: `inactive`, `standalone`, `amplified`, `redundant`, `gated`, `reversed`.
Defaults: marginal sampler, `n_samples=100`, `tau_label=0.1` (fraction of sd of the output), `kappa=0.1`.
The explained output is the prediction (regression) or the probability of the predicted class (classification); true labels are not used.

```python
import local_cbfi_v2 as V2
res = V2.decompose_instance(model, X_test.iloc[0], X_train.sample(1000), task="reg", n_samples=200)
print(res[["G1", "G4", "Imp", "label", "label_conf"]])
```

## Experiments (`revision_exp/`)
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
Models are scikit-learn pipelines (StandardScaler -> estimator) fitted on the training split only.

## Data
Wine quality, medical insurance and bike sharing are included in `dataset/`. Download the others into `dataset/`:
Adult (`adult.data.csv`, UCI), Bank Marketing (`bank.csv`, UCI), Diabetes Health Indicators
(`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`, Kaggle), Ames Housing (`AmesHousing.csv`).
California Housing is fetched through scikit-learn.

## Environment
Python 3.10, see `requirements_server.txt` (scikit-learn 1.7.1, shap 0.49.1, shapiq 1.4.1, lime 0.2.0.1,
xgboost 3.0.5, catboost 1.2.10, pytorch-tabnet 4.1.0). Experiments were run on 24 CPU cores with an RTX 3090.
