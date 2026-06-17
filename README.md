# Beyond Feature Attribution: Recovering Latent Interaction Structures via Localized CBFI

![Framework](images/Fig_01.png)

An official Python implementation of **Localized Case-Based Feature Importance (CBFI)**, a model-agnostic explainable AI (XAI) framework designed to expose the latent interaction structures embedded within complex machine learning models. 

Instead of merely distributing numerical "payouts" like traditional additive frameworks (SHAP, LIME), Localized CBFI shifts the explainable AI paradigm from simple importance attribution to the **recovery of interaction-driven structures**.

---

## 📝 Description

Localized CBFI overcomes the structural limitations of existing additive explanation models by explicitly decomposing feature interactions at the individual instance level. Traditional methods often absorb complex, non-linear interactions into single feature scores, smoothing over the structural logic that governs the model's decision manifold. 

This framework structurally isolates independent **Main Effects ($G_1$)** from pure, context-dependent **Synergistic Interactions ($G_4$)** at the individual instance level. It provides objective, high-resolution diagnostic maps of a model's internal decision mechanisms, making it vital for high-stakes domains where understanding the internal logic is as essential as predictive accuracy.

![Important chart](images/Fig_02.png)

---

## 📊 Dataset Information

The experimental pipeline integrates public datasets characterized by distinct domain attributes and complex nonlinear relationships:

1. **Wine Quality Dataset (Cortez et al., 2009)**: Consists of red wine samples with 11 chemical properties (acidity, residual sugar, alcohol, etc.). It is ideal for analyzing regulatory mechanisms ($G_4$) where final grades are determined by the delicate balance and interdependence between multiple components rather than single independent marginal variables.
2. **Medical Insurance Dataset (Lantz, 2013)**: Predicts annual medical charges based on individual attributes (age, BMI, smoking status). It features a strong nonlinear dependency structure where the combination of specific risk factors creates exponential synergies.
3. **Multi-domain Benchmarks**: Extensively validated across various high-stakes domains including finance, healthcare, and real estate (e.g., Ames Housing, Adult, Bike Sharing, and Diabetes datasets).

---

## 💻 Code Information

The core backend pipeline resides in `local_cbfi_clean_20260410.py` and provides a comprehensive explanation engine along with advanced visualization suites:

* **Core Explanation Engines**: 
  * `explain_local_cbfi_classification_conditional()`: Decomposes instance-level classification predictions into structural groups using conditional permutation sampling.
  * `explain_local_cbfi_regression_conditional()`: Restores regression error-reduction trajectories by reflecting the structural decision tree logic.
* **Neighborhood Estimators**: 
  * `_get_conditional_samples()`: Extracts a localized neighborhood sample pool based on feature correlations and KNN to preserve the learned data manifold.
* **Interaction Mapping**: 
  * `get_local_pairwise_interaction()` & `generate_all_interactions()`: Evaluate structural interaction topologies across unique feature pairs.
* **Visualization Tools**: Includes high-resolution plotting wrappers for bar attribution summaries (`visualize_feature_contribution`), 1:1 local dependencies (`visualize_local_pairwise_interaction`), and network structures (`visualize_feature_interaction_graph`, `visualize_draggable_interaction_graph`).

---

## ⚙️ Requirements

The entire framework and experimental pipelines are implemented using Python (v3.14+). Ensure the following dependencies are installed:

```bash
numpy >= 2.4.3
pandas >= 2.0.0
scikit-learn >= 1.8.0
networkx >= 3.0
matplotlib >= 3.10.8
```

## 🚀 Usage Instructions

The library and experimental code for Localized CBFI are available at https://github.com/dkumango/Local_CBFI/. You can clone the repository and import the module directly:Basic Structural Decomposition TablePythonimport pandas as pd
from sklearn.ensemble import RandomForestClassifier
from local_cbfi_clean_20260410 import get_cbfi_table, generate_all_interactions

1. Prepare your data and model
X_train, y_train, background_data = ...
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

2. Select a target instance to explain
instance = background_data.iloc[0]

3. Generate Localized CBFI structural diagnostic table
cbfi_table = get_cbfi_table(
    model=model, 
    instance=instance, 
    background_data=background_data, 
    n_samples=100, 
    job_type='classification'
)
print(cbfi_table)
Mapping Pairwise Structural GraphsPython# Generate interaction data for network graphs
interaction_df = generate_all_interactions(
    model=model, 
    instance=instance, 
    background_data=background_data, 
    n_samples=100, 
    job_type='classification'
)

## 🔬 Methodology

Localized CBFI partitions instance-level feature contributions into distinct, mutually exclusive structural components:Main Effect ($G_1$): The independent drive of a feature, isolated from any variable coupling.Interaction ($G_4$): The pure synergy or regulatory suppression that emerges only through the joint presence of multiple features.$G_4 > 0$: Synergistic Interaction (cooperative feature reinforcement).$G_4 < 0$: Regulatory/Suppressive Interaction (contextual attenuation).Case-Based Sampling StrategyTo solve the combinatorial explosion problem common in game-theoretic approaches, Localized CBFI replaces exhaustive search with a case-based permutation sampling policy. By shuffling target features dynamically against background manifolds instead of using static references, it maintains linear scalability with respect to feature dimensions, achieving sub-second runtimes (<1.0s) per instance.📈 Latent Interaction GraphBy aggregating pairwise $G_4$ strength across instances, Localized CBFI explicitly maps out how predictive signals propagate through interconnected feature relationships, uncovering the hidden structural topology of black-box architectures.

## 📜 Citations

If you use this framework, code, or associated datasets in your research, please cite the following foundational work:코드 스니펫@article{oh2022predictive,

  title={Predictive case-based feature importance and interaction},
  author={Oh, Sejong},
  journal={Information Sciences},
  volume={593},
  pages={155--176},
  year={2022},
  publisher={Elsevier}
}

@article{oh2026beyond,
  title={Beyond Feature Attribution: Recovering Latent Interaction Structures in Machine Learning Models via Localized CBFI},
  author={Oh, Sejong},
  journal={PeerJ Computer Science},
  year={2026}
}

## 📄 License & Contribution 

GuidelinesLicenseThis project is licensed under the MIT License - see the LICENSE file for details.Contribution GuidelinesWe welcome contributions to optimize high-dimensional scalability and extend compatibility toward advanced tabular deep learning models.Fork the Repository at https://github.com/dkumango/Local_CBFI/.Create a feature branch (git checkout -b feature/AmazingFeature).Commit your changes and push to the branch.Open a Pull Request for code review.
