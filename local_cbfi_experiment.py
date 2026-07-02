#############################################################
# Local CBFI Experiment with Wine Quality and Insurance Cost Datasets
# Compare with Other methods
# 2026-06-03    
#############################################################

import local_cbfi_clean_20260410 as CBFI 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

##############################################################################
##############################################################################
# CLASSIFICATION
##############################################################################
##############################################################################


def load_and_preprocess_wine_data(random_seed=42):
    """
    Loads the Wine Quality dataset, removes duplicates, splits the data into training 
    and testing sets, and applies standard scaling which is essential for stable 
    KNN distance computations in the CBFI conditional sampling process.
    """    
    df = pd.read_csv("dataset/winequality-red.csv") 
    df.drop_duplicates(inplace=True)                   #  Remove duplicates
    
    # Prepare features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split data into train/test sets (8:2 split to ensure sufficient samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # Feature scaling using StandardScaler
    # Essential step for KNN computation in CBFI's _get_conditional_samples()
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Reset indices for easier iloc access later)
    X_train_scaled.reset_index(drop=True, inplace=True)
    X_test_scaled.reset_index(drop=True, inplace=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Execute and verify data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_wine_data()
print(f"Dataset Loaded: {len(X_train)} train samples, {len(X_test)} test samples")

# RF model build
model_wine = RandomForestClassifier(n_estimators=50,  random_state=42)
model_wine.fit(X_train, y_train)
# model_wine.score(X_test, y_test)
print(f"Random Forest Test Accuracy: {model_wine.score(X_test, y_test):.4f}")

##################
## CBFI TEST #####
##################


pick = 8  # Zero-based index, e.g., Sample #18 (quality: 7)
sample_instance = X_test.iloc[pick]   # quality:7
actual_1 = y_test[pick]
predicted_1 = model_wine.predict(pd.DataFrame([sample_instance]))[0]
print(f"Sample Index: {pick}, Actual Quality: {actual_1}, Predicted Quality: {predicted_1}")

df_cbfi_table_1 = CBFI.get_cbfi_table(model_wine, sample_instance, X_train, n_samples=100, job_type='classification')
print(df_cbfi_table_1)
CBFI.visualize_feature_contribution(df_cbfi_table_1, sample_instance, actual_1, predicted_1, scaler)

print(f"Analyzing 1:1 interactions for feature:{X_train.columns[10]}") # alcohol
df_interact_1 = CBFI.get_local_pairwise_interaction(model_wine, sample_instance, X_train, X_train.columns[10], n_samples=100, 
                                             job_type='classification')
CBFI.visualize_local_pairwise_interaction(df_interact_1, sample_instance, actual_1, predicted_1, scaler)

print(f"Analyzing 1:1 interactions for feature:{X_train.columns[5]}") # free sulfur dioxide
df_interact_12 = CBFI.get_local_pairwise_interaction(model_wine, sample_instance, X_train, X_train.columns[5], n_samples=100, 
                                             job_type='classification')
CBFI.visualize_local_pairwise_interaction(df_interact_12, sample_instance, actual_1, predicted_1, scaler)

## interaction network graph
importance_df_1 = df_cbfi_table_1['Total']
interaction_df_1 = pd.read_csv("interaction_df_1.csv")  # # Load previously computed interaction data

# generate interaction data and save to csv
# interaction_df_1 = CBFI.generate_all_interactions(model_wine, sample_instance, X_train, n_samples=100, job_type='classification')
# CBFI.visualize_feature_interaction_graph(importance_df_1, interaction_df_1, threshold=0.01) # static
# interaction_df_1.to_csv("interaction_df_1.csv", index=False)

CBFI.visualize_draggable_interaction_graph(importance_df_1, interaction_df_1, threshold=0.001) # interactive

#############
## SHAP #####
#############

import shap

# 1. Initialize SHAP Explainer (Tree-specific)
explainer_shap = shap.TreeExplainer(model_wine)

# 2. Calculate SHAP values for the test instance
# For multi-class, shap_values is a list of shape [num_classes, num_samples, num_features]
shap_values = explainer_shap.shap_values(sample_instance)

# 3.Visualize results for the predicted class of the specific sample
# Find the index of the predicted class (e.g., if prediction is 6, find its index in the class list)
pred_class_idx = list(model_wine.classes_).index(predicted_1)

print(f"SHAP Analysis for Predicted Class: {predicted_1}")
#shap.initjs() # Required in Jupyter Notebook environments
shap.force_plot(
    explainer_shap.expected_value[pred_class_idx], 
    shap_values[:,pred_class_idx],          # feature importance for the predicted class
    sample_instance,
    matplotlib=True
)
# SHAP feature importance
SHAP_imp = pd.Series(shap_values[:, pred_class_idx], index=X_train.columns)
CBFI_imp = df_cbfi_table_1['Total']
CBFI_imp = CBFI_imp.reindex(sample_instance.index)  # Align feature order identically with SHAP
SHAP_imp.corr(CBFI_imp)
CBFI_imp.corr(np.abs(SHAP_imp))

# feature interaction
# Caution: Computation takes exponentially longer than standard SHAP values.
interaction_values = explainer_shap.shap_interaction_values(X_test.iloc[pick:pick+1])

# Result shape: (num_samples, num_features, num_features)
# Verify the 11x11 interaction matrix for sample #18
sample_interaction = interaction_values[0][10,:,pred_class_idx]

# Calculate correlation between CBFI G4 and SHAP interactions
G4 = df_interact_1.iloc[:,0].to_list()            # Must restore original feature order
SHAP_interact = sample_interaction[:10].tolist()  # Interaction values excluding the diagonal (exclude self-interaction)

correlation = np.corrcoef(G4, SHAP_interact)[0, 1]
print(f"Correlation between CBFI G4 and SHAP Interaction Values: {correlation:.4f}")

###########
## LIME
############

from lime import lime_tabular

# 1. Initialize LIME Explainer
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=model_wine.classes_.astype(str).tolist(),
    mode='classification'
)

# 2. Generate explanation for the test instance
# num_features determines how many top features to display

class_idx = list(model_wine.classes_).index(predicted_1)
exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[pick].values, 
    predict_fn=model_wine.predict_proba,
    num_features=X_test.shape[1],  # Show all features
    labels=[class_idx]             # Request explanation for the predicted class
)

# 3.  Visualize results
# exp.show_in_notebook(show_table=True)
# Or check as a list

print(exp.as_list(label=class_idx ))
local_exp = exp.local_exp[class_idx]

LIME_imp = pd.Series({
    explainer_lime.feature_names[idx]: val for idx, val in local_exp
}, name='importance')

LIME_imp = LIME_imp.reindex(sample_instance.index)

CBFI_imp.corr(np.abs(LIME_imp))
LIME_imp.corr(SHAP_imp)

#############################################################################
# Fidelity Test
#############################################################################
import matplotlib.pyplot as plt

def perform_fidelity_test(model, instance, importance_dict, background_data, n_steps=5):
    """
    Measure the drop in prediction probability (Fidelity) as features are sequentially removed.
    """
    import pandas as pd
    import numpy as np

    feature_names = instance.index
    instance_df = pd.DataFrame([instance])
    
    # 1. Verify model's predicted label
    target_label = model.predict(instance_df)[0]
    
    # 2. Convert label value (e.g., 7) to probability array index (e.g., 4) (Key fix)
    target_idx = list(model.classes_).index(target_label)
    
    # 3. Extract initial probability value
    initial_proba = model.predict_proba(instance_df)[0][target_idx]
    
    reference_values = background_data.median()
    results = {}
    
    for method_name, importance_values in importance_dict.items():
        # Sort features by importance
        sorted_features = pd.Series(importance_values, index=feature_names).abs().sort_values(ascending=False).index
        
        probas = [initial_proba]
        temp_instance = instance.copy()
        
        # Sequentially remove top n_steps features
        for i in range(min(n_steps, len(sorted_features))):
            feat_to_remove = sorted_features[i]
            temp_instance[feat_to_remove] = reference_values[feat_to_remove]
            
            # Extract probability using the converted index (target_idx) each time
            new_proba = model.predict_proba(pd.DataFrame([temp_instance]))[0][target_idx]
            probas.append(new_proba)
            
        results[method_name] = probas
        
    return results

# Prepare importance data
importance_dict = {
    'SHAP': SHAP_imp, # Raw SHAP (abs processed within the function)
    'LIME': LIME_imp, 
    'CBFI': CBFI_imp  # Sum of G1 + G4
}

# Execute experiment
fidelity_results = perform_fidelity_test(model_wine, X_test.iloc[pick], importance_dict, X_train)

# Visualization
plt.figure(figsize=(10, 6))
for method, scores in fidelity_results.items():
    plt.plot(range(len(scores)), scores, marker='o', label=method)

plt.title("Fidelity Test: Prediction Probability Drop (Sample #18)")
plt.xlabel("Number of Features Removed (Top-k)")
plt.ylabel(f"Probability of Class 7")
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

######################################################################
# Synergy Removal Test
######################################################################
def perform_synergy_test(model, instance, cbfi_df, shap_df, background_data):
    """
     Measure the synergy effect resulting from the simultaneous removal of interacting feature pairs.
    """
    feature_names = instance.index
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    initial_proba = model.predict_proba(pd.DataFrame([instance]))[0][target_idx]
    
    # Set replacement values (medians)
    ref_vals = background_data.median()
    
    # 1. Select feature pairs to test
    # CBFI: Top 2 features with the highest G4 (Interaction)
    top_g4_pair = cbfi_df['Interact (G4)'].sort_values(ascending=False).index[:2].tolist()
    
    # SHAP: Top 2 features with the highest absolute contribution
    top_shap_pair = shap_df.abs().sort_values(ascending=False).index[:2].tolist()
    
    pairs = {'CBFI (G4 Pair)': top_g4_pair, 'SHAP (Top Pair)': top_shap_pair}
    results = []

    for label, pair in pairs.items():
        fx, fy = pair[0], pair[1]
        
        # Generate samples for each scenario
        inst_x = instance.copy(); inst_x[fx] = ref_vals[fx] # Fx 제거
        inst_y = instance.copy(); inst_y[fy] = ref_vals[fy] # Fy 제거
        inst_xy = instance.copy(); inst_xy[fx] = ref_vals[fx]; inst_xy[fy] = ref_vals[fy] # 둘 다 제거
        
        # easure probabilities
        p_x = model.predict_proba(pd.DataFrame([inst_x]))[0][target_idx]
        p_y = model.predict_proba(pd.DataFrame([inst_y]))[0][target_idx]
        p_xy = model.predict_proba(pd.DataFrame([inst_xy]))[0][target_idx]
        
        # Calculate probability drop
        drop_x = initial_proba - p_x
        drop_y = initial_proba - p_y
        drop_joint = initial_proba - p_xy
        
        results.append({
            'Method': label,
            'Pair': f"{fx} & {fy}",
            'Individual_Sum': drop_x + drop_y,
            'Joint_Drop': drop_joint,
            'Synergy_Index': drop_joint / (drop_x + drop_y + 1e-9)
        })

    return pd.DataFrame(results)

# --- Execution Example ---
# Execute matching the dataframe variable names.
# cbfi_df: Table containing G1~G4, shap_imp: SHAP value series
synergy_results = perform_synergy_test(model_wine, X_test.iloc[pick], df_cbfi_table_1, SHAP_imp, X_train)
print(synergy_results)

#########################################################################
#Check individual Drop values for alcohol and total sulfur dioxide
#########################################################################
def check_individual_drops(model, instance, background_data):
    """
    Measure the drop in prediction probability when specific features are individually removed.
    """
    feature_names = instance.index
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    initial_proba = model.predict_proba(pd.DataFrame([instance]))[0][target_idx]

    # Reference replacement values (medians of training data)
    ref_vals = background_data.median()

    features_to_check = ['alcohol', 'total sulfur dioxide']
    results = {}

    for feat in features_to_check:
        temp_instance = instance.copy()
        temp_instance[feat] = ref_vals[feat]
        
        # Calculate probability after removing feature
        new_proba = model.predict_proba(pd.DataFrame([temp_instance]))[0][target_idx]
        drop_val = initial_proba - new_proba
        
        results[feat] = {
            'Original_Proba': initial_proba,
            'New_Proba': new_proba,
            'Drop': drop_val
        }

    return pd.DataFrame(results).T

# Execution (based on pick=18 sample)
individual_drops = check_individual_drops(model_wine, X_test.iloc[pick], X_train)
print(individual_drops)

#########################################################################
# Experiment: Plot probability curve (Partial Dependence style) based on 
# SO2 presence while varying alcohol levels
#########################################################################
def plot_interaction_sensitivity(model, instance, background_data, target_feat='alcohol', interact_feat='total sulfur dioxide'):
    """
    Visualize the regulatory effect of an interaction feature (SO2) according to changes in the target feature (alcohol).
    """
    # 1. Preparation
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    ref_val = background_data[interact_feat].median()
    
    # 2. Set range for target feature (min to max in dataset)
    x_range = np.linspace(background_data[target_feat].min(), background_data[target_feat].max(), 100)
    
    proba_with_so2 = []
    proba_without_so2 = []
    
    for val in x_range:
        # Scenario A: Maintain original interaction feature state
        inst_a = instance.copy()
        inst_a[target_feat] = val
        p_a = model.predict_proba(pd.DataFrame([inst_a]))[0][target_idx]
        proba_with_so2.append(p_a)
        
        # Scenario B: Remove interaction feature (replace with median)
        inst_b = instance.copy()
        inst_b[target_feat] = val
        inst_b[interact_feat] = ref_val
        p_b = model.predict_proba(pd.DataFrame([inst_b]))[0][target_idx]
        proba_without_so2.append(p_b)
        
    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, proba_with_so2, label=f'With {interact_feat} (Original)', color='teal', linewidth=2)
    plt.plot(x_range, proba_without_so2, label=f'Without {interact_feat} (Median)', color='salmon', linestyle='--', linewidth=2)
    
    # Indicate the actual target feature value of the current instance
    plt.axvline(x=instance[target_feat], color='gray', linestyle=':', label=f'Current {target_feat} ({instance[target_feat]})')
    
    plt.title(f"Interaction Effect: How '{interact_feat}' regulates '{target_feat}'", fontsize=14)
    plt.xlabel(f"{target_feat} Content", fontsize=12)
    plt.ylabel(f"Probability of Class {target_label}", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Execution
plot_interaction_sensitivity(model_wine, X_test.iloc[pick], X_train)


##############################################################################
##############################################################################
# REGRESSION
##############################################################################
##############################################################################

def load_and_preprocess_insurance_data(random_seed=42):
    """
    Loads the Medical Insurance Cost dataset, encodes categorical variables numerically,
    splits the data into training and testing sets, and applies standard scaling 
    to ensure stable KNN conditional sampling for CBFI interaction mapping.
    """
    df = pd.read_csv("dataset/Medical_Cost.csv")
    
    # 2. Encode categorical variables
    le = LabelEncoder()
    # smoker: yes -> 1, no -> 0 (critical for interaction analysis)
    df['smoker'] = le.fit_transform(df['smoker'])
    # sex: male, female conversion
    df['sex'] = le.fit_transform(df['sex'])
    # rregion: convert 4 regions
    df['region'] = le.fit_transform(df['region'])
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # 3. Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    # 4. Scaling (ensures stability of KNN conditional sampling)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Execution
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_insurance_data()

# RF model build
model_insurance = RandomForestRegressor(n_estimators=150,  random_state=42)   
model_insurance.fit(X_train, y_train)
# model_insurance.score(X_test, y_test)
print(f"Random Forest Test R2: {model_insurance.score(X_test, y_test):.4f}")


#############
## CBFI #####
#############

pick = 6  # min : 6, max : 14
sample_instance = X_test.iloc[pick]   # quality:7
actual_2 = y_test.iloc[pick]
predicted_2 = model_insurance.predict(pd.DataFrame([sample_instance]))[0]
print(f"Sample Index: {pick}, Actual Insurance Cost: {actual_2}, Predicted Insurance Cost: {predicted_2}")

df_cbfi_table_2 = CBFI.get_cbfi_table(model_insurance, sample_instance, X_train, actual_2, predicted_2,n_samples=100, job_type='regression')
print(df_cbfi_table_2)
CBFI.visualize_feature_contribution(df_cbfi_table_2, sample_instance, actual_2, predicted_2, scaler)

print(f"Analyzing 1:1 interactions for feature:{X_train.columns[4]}") # smoker
df_interact_2 = CBFI.get_local_pairwise_interaction(model_insurance, sample_instance, X_train, X_train.columns[4], n_samples=100, 
                                             job_type='regression')
CBFI.visualize_local_pairwise_interaction(df_interact_2, sample_instance, actual_2, predicted_2, scaler)

## interaction network graph
importance_df_2 = df_cbfi_table_2['Total']
interaction_df_2 = pd.read_csv("interaction_df_2_insurance.csv") # Load previously computed interaction data

# Generate interaction data and save to csv
#interaction_df_2 = CBFI.generate_all_interactions(model_insurance, sample_instance, X_train, n_samples=100, job_type='regression')
#interaction_df_2.to_csv("interaction_df_2_insurance.csv", index=False)

CBFI.visualize_draggable_interaction_graph(importance_df_2, interaction_df_2, threshold=100) # interactive


#############
## SHAP #####
#############

import shap

# 1. Initialize SHAP Explainer (Regression)
explainer_shap = shap.TreeExplainer(model_insurance)

# 2. Calculate SHAP values for the test instance
# For regression, shap_values is a single array of shape [num_samples, num_features]
shap_values = explainer_shap.shap_values(sample_instance)

# 3. Visualize specific sample (no class selection needed for regression)
print(f"SHAP Analysis for Predicted Insurance Charges")

base_val = explainer_shap.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[0]

# 4. Extract Data values (handle Series vs DataFrame)
# If sample_instance is a Series, use .values; if DataFrame, use .iloc[0].values
if isinstance(sample_instance, pd.Series):
    actual_data = sample_instance.values
else:
    actual_data = sample_instance.iloc[0].values

# 5. Create Explanation object
exp = shap.Explanation(
    values=shap_values[0],          # SHAP contribution array
    base_values=base_val,           # Model's base/expected value
    data=actual_data,               # Actual feature values
    feature_names=X_train.columns.tolist()
)

# 6.  Extract SHAP Feature Importance and calculate correlation
# In regression, shap_values[0] simply contains the contributions of each feature
SHAP_imp = pd.Series(shap_values, index=X_train.columns)
CBFI_imp = df_cbfi_table_2['Total']
CBFI_imp = CBFI_imp.reindex(SHAP_imp.index) # 특징 순서 정렬

corr_raw = SHAP_imp.corr(CBFI_imp)
corr_abs = CBFI_imp.corr(np.abs(SHAP_imp))

print(f"Correlation (Raw): {corr_raw:.4f}")
print(f"Correlation (Absolute): {corr_abs:.4f}")

# 7. Calculate SHAP Interaction Values
# Result shape: (num_samples, num_features, num_features)
interaction_values = explainer_shap.shap_interaction_values(sample_instance)
# Extract [num_features x num_features] interaction matrix for the sample
matrix_interaction = interaction_values 

# 8. Extract SHAP Interaction values for a specific variable (e.g., feat_x)
# If feat_x is at index 0 (e.g., age), fetch that row
feat_x_idx = 4 # 예시로 'smoker' 변수의 인덱스  
shap_interact_row = matrix_interaction[feat_x_idx]

# 9. Calculate correlation between CBFI G4 and SHAP Interaction
# Match with CBFI's G4 results excluding the diagonal (self-interaction) of SHAP matrix
# df_interact_2 must be a table containing G4 values between feat_x and all other feat_y
G4_values = df_interact_2['Interaction'].tolist()

# Extract feat_x row from SHAP interaction matrix, excluding itself (diagonal
mask = np.ones(len(shap_interact_row), dtype=bool)
mask[feat_x_idx] = False
SHAP_interact_filtered = shap_interact_row[mask].tolist()

correlation_g4 = np.corrcoef(G4_values, SHAP_interact_filtered)[0, 1]
print(f"Correlation between CBFI G4 and SHAP Interaction: {correlation_g4:.4f}")
# Result:  0.2430


###########
## LIME
############

from lime import lime_tabular

# 1. Initialize LIME Explainer
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    mode='regression'
)

# 2. Generate explanation for the test instance
# num_features determines how many top features to display
exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[pick].values, 
    predict_fn=model_insurance.predict,
    num_features=X_test.shape[1]         # Show all features
)

# 3.  Visualize results
# exp.show_in_notebook(show_table=True)
# Or check as a list

print("--- LIME Feature Importance List ---")
print(exp.as_list())

# LIn LIME regression mode, results are usually stored at index 1.
# If an error occurs, fallback to list(exp.local_exp.keys())[0] for flexibility.
target_key = list(exp.local_exp.keys())[0]
local_exp = exp.local_exp[target_key]

# Match feature names and create Series
LIME_imp = pd.Series({
    explainer_lime.feature_names[idx]: val for idx, val in local_exp
}, name='importance')

#  Align feature order identically with SHAP and CBFI
LIME_imp = LIME_imp.reindex(sample_instance.index)

#  Calculate correlations
# Validity check with CBFI (absolute value basis)
corr_cbfi_lime = CBFI_imp.corr(np.abs(LIME_imp))
# Consistency check with SHAP (comparison between additive models)  
corr_lime_shap = LIME_imp.corr(SHAP_imp)

print(f"Correlation (CBFI vs |LIME|): {corr_cbfi_lime:.4f}")
print(f"Correlation (LIME vs SHAP): {corr_lime_shap:.4f}")

#############################################################################
# Fidelity Test (Regression)
#############################################################################
import matplotlib.pyplot as plt

def perform_fidelity_test_regression(model, instance, importance_dict, background_data, n_steps=5):
    """
    Measure the shift in predicted regression value as features are sequentially removed.
    """
    feature_names = instance.index
    instance_df = pd.DataFrame([instance])
    
    # Extract initial prediction (in regression, the value itself is the result)
    initial_pred = model.predict(instance_df)[0]
    
    # Reference replacement values when removing features (training data medians)
    reference_values = background_data.median()
    results = {}
    
    for method_name, importance_values in importance_dict.items():
        # Sort features by importance (absolute basis)
        sorted_features = pd.Series(importance_values, index=feature_names).abs().sort_values(ascending=False).index
        
        preds = [initial_pred]
        temp_instance = instance.copy()
        
        # Sequentially remove top n_steps features (replace with median)
        for i in range(min(n_steps, len(sorted_features))):
            feat_to_remove = sorted_features[i]
            temp_instance[feat_to_remove] = reference_values[feat_to_remove]
            
            # Calculate new prediction after feature removal
            new_pred = model.predict(pd.DataFrame([temp_instance]))[0]
            preds.append(new_pred)
            
        results[method_name] = preds
        
    return results


# 1. Prepare importance data
importance_dict = {
    'SHAP': SHAP_imp, # Raw SHAP (abs processed within the function)
    'LIME': LIME_imp, 
    'CBFI': CBFI_imp  # Sum of G1 + G4
}

# 2. Execute experiment
fidelity_results = perform_fidelity_test_regression(model_insurance, X_test.iloc[pick], importance_dict, X_train)

# 3. Visualization
plt.figure(figsize=(10, 6))
for method, scores in fidelity_results.items():
    plt.plot(range(len(scores)), scores, marker='o', label=method)

plt.title("Fidelity Test: Prediction Probability Drop (Sample #6)")
plt.xlabel("Number of Features Removed (Top-k)")
plt.ylabel(f"Predicted Charge")
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

######################################################################
# Synergy Removal Test
######################################################################

def perform_synergy_test_regression(model, instance, cbfi_df, shap_df, background_data):
    """
    Measure the prediction value change (synergy) caused by the simultaneous removal of interacting feature pairs in regression.
    """
    feature_names = instance.index
    # 1. Extract initial prediction (regression result)
    initial_pred = model.predict(pd.DataFrame([instance]))[0]
    
    # Set replacement values (medians)
    ref_vals = background_data.median()
    
    # 2. Select feature pairs to test
    # CBFI: Top 2 features with the highest G4 (Interaction)
    top_g4_pair = cbfi_df['Interact (G4)'].sort_values(ascending=False).index[:2].tolist()
    
    # SHAP: Top 2 features with highest absolute contribution (based on regression SHAP_imp)
    top_shap_pair = shap_df.abs().sort_values(ascending=False).index[:2].tolist()
    
    pairs = {'CBFI (G4 Pair)': top_g4_pair, 'SHAP (Top Pair)': top_shap_pair}
    results = []

    for label, pair in pairs.items():
        fx, fy = pair[0], pair[1]
        
        # Generate samples for each scenario
        inst_x = instance.copy(); inst_x[fx] = ref_vals[fx] # Fx 제거
        inst_y = instance.copy(); inst_y[fy] = ref_vals[fy] # Fy 제거
        inst_xy = instance.copy(); inst_xy[fx] = ref_vals[fx]; inst_xy[fy] = ref_vals[fy] # 둘 다 제거
        
        # Measure prediction (use predict instead of predict_proba)
        pred_x = model.predict(pd.DataFrame([inst_x]))[0]
        pred_y = model.predict(pd.DataFrame([inst_y]))[0]
        pred_xy = model.predict(pd.DataFrame([inst_xy]))[0]
        
        # Calculate shift (Drop): difference between initial and post-removal values
        drop_x = initial_pred - pred_x
        drop_y = initial_pred - pred_y
        drop_joint = initial_pred - pred_xy
        
        # Calculate synergy index
        synergy_index = drop_joint / (drop_x + drop_y + 1e-9)
        
        results.append({
            'Method': label,
            'Pair': f"{fx} & {fy}",
            'Individual_Sum': drop_x + drop_y,
            'Joint_Drop': drop_joint,
            'Synergy_Index': synergy_index
        })

    return pd.DataFrame(results)

# --- Execution Example ---
synergy_results = perform_synergy_test_regression(model_insurance, X_test.iloc[pick], df_cbfi_table_2, SHAP_imp, X_train)
print(synergy_results)
#             Method          Pair  Individual_Sum   Joint_Drop  Synergy_Index
# 0   CBFI (G4 Pair)  smoker & age    -6646.155188 -6646.155188            1.0
# 1  SHAP (Top Pair)  age & smoker    -6646.155188 -6646.155188            1.0



