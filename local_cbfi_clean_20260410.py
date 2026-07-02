#######################################################################################
# Local CBFI with conditional permutation
#######################################################################################

from pyexpat import features, model

from matplotlib.pylab import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################################################################################
from sklearn.neighbors import NearestNeighbors

def _get_conditional_samples(instance, background_data, target_feature, n_neighbors=20):
    """
    Extracts a localized neighborhood sample pool based on feature correlations and K-Nearest Neighbors to preserve the data manifold.
    """
    # 1. Select contextual features based on correlation matrix
    corr_matrix = background_data.corr()
    relevant_features = corr_matrix[target_feature].abs().sort_values(ascending=False)
    cond_features = relevant_features.index[1:3].tolist()
    
    # 2. Fit KNN model
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(background_data[cond_features])
    
    query_instance = pd.DataFrame([instance[cond_features]])
    
    dist, indices = nn.kneighbors(query_instance)
    
    return background_data.iloc[indices[0]]

#########################################################################################
def explain_local_cbfi_classification_conditional(model, instance, background_data, target_feature, n_samples=100):
    """
    Decomposes classification predictions into mutually exclusive structural groups (G1 to G4) using conditional permutation sampling.
    """

    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    
    # Acquire conditional sampling pool
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature)
    
    other_features = [f for f in instance.index if f != target_feature]
    counts = {'G1': 0, 'G2': 0, 'G3': 0, 'G4': 0}

    seed=0
    for _ in range(n_samples):
        # Sample from the conditional pool instead of the full dataset (Key Modification)
        random_sample = conditional_pool.sample(1, random_state=seed).iloc[0]
        seed += 1

        ds_fx = instance.copy()
        for f in other_features: ds_fx[f] = random_sample[f]
            
        ds_fx_minus = instance.copy()
        ds_fx_minus[target_feature] = random_sample[target_feature]
        
        pred_fx = model.predict(pd.DataFrame([ds_fx]))[0]
        pred_fx_minus = model.predict(pd.DataFrame([ds_fx_minus]))[0]
        
        is_fx_correct = (pred_fx == target_label)
        is_fx_minus_correct = (pred_fx_minus == target_label)
        
        # Apply the grouping logic defined in the paper 
        if is_fx_correct and not is_fx_minus_correct: counts['G1'] += 1
        elif not is_fx_correct and is_fx_minus_correct: counts['G2'] += 1
        elif is_fx_correct and is_fx_minus_correct: counts['G3'] += 1
        elif not is_fx_correct and not is_fx_minus_correct: counts['G4'] += 1

    ratios = {k: v / n_samples for k, v in counts.items()}
    return ratios, ratios['G1'] + ratios['G4']

##############################################################################
def explain_local_cbfi_regression_conditional(model, instance, background_data, target_feature, actual, pred, n_samples=100):
    """
    Exposes G1, G2, G3, and G4 by perfectly reflecting the decision tree logic of Figure 7 in the manuscript.
    """
    pred = model.predict(pd.DataFrame([instance]))[0]
    diff_u = abs(pred - actual)

    # Conditional sampling using the previously defined _get_conditional_samples function
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature)
    other_features = [f for f in instance.index if f != target_feature]
    
    # Initialize error reduction contribution sums for each group
    sums = {'G1': 0.0, 'G2': 0.0, 'G3': 0.0, 'G4': 0.0}
    eps = 1e-9     # Threshold value for floating-point comparison

    seed=0
    for _ in range(n_samples):
        random_sample = conditional_pool.sample(1, random_state=seed).iloc[0]
        seed += 1

        # DS(Fx): Keep Fx active, permute the remaining features
        ds_fx = instance.copy()
        for f in other_features: ds_fx[f] = random_sample[f]
            
        # DS(Fx-): Permute Fx, keep the remaining features fixed
        ds_fx_minus = instance.copy()
        ds_fx_minus[target_feature] = random_sample[target_feature]
        
        pred_x = model.predict(pd.DataFrame([ds_fx]))[0]
        pred_minus = model.predict(pd.DataFrame([ds_fx_minus]))[0]
        
        diff_x = abs(pred_x - actual)
        diff_minus = abs(pred_minus - actual)
        
        # Contribution definition: Difference between baseline error (diff_minus) and final error (diff_u)
        contribution = diff_minus - diff_u
        
        # --- Apply Decision Tree Logic from Figure 7 ---
        
        # 1. G3 (Common Contribution): No change observed in error magnitude
        if abs(diff_minus - diff_u) < eps:
            sums['G3'] += contribution        # Converges mathematically to zero
            
        # 2. Positive Regime: The full model error (diff_u) is smaller than baseline error (diff_minus)
        elif diff_minus > diff_u:
            if diff_x > diff_u:
                sums['G4'] += contribution    # Positive Interaction 
            else:
                sums['G1'] += contribution    # Positive Power 
                
        # 3. Negative Regime: The full model error (diff_u) is larger than baseline error (diff_minus)
        elif diff_u > diff_minus:
            if diff_x > diff_u:
                sums['G1'] += contribution     # Negative Power 
            else:
                sums['G4'] += contribution     # Negative Interaction 
                
    # Normalize results by the total number of samples
    ratios = {
        'G1': sums['G1'] / n_samples,
        'G2': 0.0,                             # Not defined in regression models 
        'G3': sums['G3'] / n_samples,
        'G4': sums['G4'] / n_samples
    }
    
    # Aggregate total importance (Equation 21): Imp = FP + Int
    importance = ratios['G1'] + ratios['G4']
    
    return ratios, importance

########################################################################
def get_cbfi_table(model, instance, background_data, actual=None, pred=None,n_samples=100, job_type='classification'):
    """
    Computes Localized CBFI metrics for all features of an instance and returns a dataframe sorted by total importance.
    """
    results = []
    features = instance.index

    print("Calculating Local CBFI for all features...")
    for f in features:
        if job_type == 'classification':
            ratios, _ = explain_local_cbfi_classification_conditional(model, instance, background_data, f, n_samples)
        else:           # regression
            ratios, _ = explain_local_cbfi_regression_conditional(model, instance, background_data, f, actual, pred, n_samples)

        results.append({
            'Feature': f,
            'Power (G1)': ratios['G1'],
            'Others (G2)': ratios['G2'],
            'Common (G3)': ratios['G3'],
            'Interact (G4)': ratios['G4']
        })
    
    df_cbfi_table = pd.DataFrame(results).set_index('Feature')

    # Sort features by total structural importance (G1 + G4) 
    df_cbfi_table['Total'] = df_cbfi_table['Power (G1)'] + df_cbfi_table['Interact (G4)']
    df_cbfi_table = df_cbfi_table.sort_values(by='Total', ascending=True)

    return df_cbfi_table


########################################################################################
## Interaction Analysis
########################################################################################
def _local_pairwise_interaction_regression(model, instance, background_data, feat_x, 
                                            n_neighbors=50, 
                                            n_samples=500, 
                                            random_state=42):
    """
    Computes 1:1 pairwise interaction mapping table between feat_x and other features in regression models. 
    """
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_y = model.predict(instance_df)[0]

    interaction_results = []

    # 1. Common Utility: Compute diff vector using a fixed sampling pool
    def get_diff_vectorized(fixed_feats, sampled_pool):
        diffs = []
        for i in range(n_samples):
            random_sample = sampled_pool.iloc[i]
            temp_ds = instance.copy()

            # Retrieve from sampled data unless the features are explicitly fixed
            for f in features:
                if f not in fixed_feats:
                    temp_ds[f] = random_sample[f]
            
            pred = model.predict(pd.DataFrame([temp_ds]))[0]
            diffs.append(abs(pred - target_y))
        return np.array(diffs)

    # Loop over other features to compute pairwise interaction effects
    for feat_y in features:
        if feat_x == feat_y: continue
        
        # 2. Extract localized neighborhood based on {Fx, Fy} (Maintains structural consistency with classification)
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        
        # 3. Pre-sample with replacement from the neighborhood pool to ensure reproducibility
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        # 4. Calculate scenario-specific predictive error (diff vectors)
        diff_x = get_diff_vectorized([feat_x], sampled_pool)
        diff_y = get_diff_vectorized([feat_y], sampled_pool)
        diff_xy = get_diff_vectorized([feat_x, feat_y], sampled_pool)
        
        # 5. Apply Equation 22 from the manuscript: Mean error reduction value
        int_val = (np.mean(diff_x - diff_xy) + np.mean(diff_y - diff_xy)) / 2
        
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': int_val})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False) 
 

########################################################################
def _local_pairwise_interaction_classification(model, instance, background_data, feat_x, 
                                               n_neighbors=50, # Preserve locality (smaller value)
                                               n_samples=500,  # Statistical stability (larger value)
                                               random_state=42):
    """
    Analyzes conditional 1:1 pairwise interaction between feat_x and other variables based on Section 2.3 and Eq 14.
    """
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    
    interaction_results = []
    
    for feat_y in features:
        if feat_x == feat_y: continue
        
        # 1. Establish conditional sampling pool
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        # 2. Create a fixed-size neighborhood pool to capture local manifold structure
        nn = NearestNeighbors(n_neighbors=n_neighbors) 
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        
        # 3. Perform bootstrapping (500 iterations) within the local pool for stability and consistency
        # replace=True is mandatory as n_samples exceeds the raw pool size
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        g4_count = 0
        
        for i in range(n_samples):
            random_sample = sampled_pool.iloc[i]         # Access samples using a reproducible deterministic sequence
            
            # Construct evaluation datasets)
            ds_xy, ds_x, ds_y = instance.copy(), instance.copy(), instance.copy()
            
            for f in features:
                if f not in [feat_x, feat_y]:
                    ds_xy[f] = random_sample[f]
                if f != feat_x:
                    ds_x[f] = random_sample[f]
                if f != feat_y:
                    ds_y[f] = random_sample[f]
            
            # Execute model predictions
            pred_xy = model.predict(pd.DataFrame([ds_xy]))[0]
            pred_x = model.predict(pd.DataFrame([ds_x]))[0]
            pred_y = model.predict(pd.DataFrame([ds_y]))[0]
            
            if (pred_xy == target_label) and (pred_x != target_label) and (pred_y != target_label):
                g4_count += 1
        
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': g4_count / n_samples})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False)


########################################################################
def get_local_pairwise_interaction(model, instance, background_data, feat_x, n_samples=100, job_type='classification'):
    """
    Routes to the appropriate pairwise interaction function based on whether the task is classification or regression.
    """
    if job_type == 'classification':
        return _local_pairwise_interaction_classification(model, instance, background_data, feat_x, n_samples)
    else:            # regression
        return _local_pairwise_interaction_regression(model, instance, background_data, feat_x,  n_samples=n_samples)

########################################################################
from itertools import combinations

def generate_all_interactions(model, instance, background_data, n_samples=100, job_type='classification'):
    """
    Computes structural interaction weights across all unique feature pairs to construct a network graph.
    """
    features = instance.index.tolist()
    interaction_list = []
    
    # 1. Generate all pairwise feature combinations
    feature_pairs = list(combinations(features, 2))
    
    print(f"Total {len(feature_pairs)} pairs to analyze...")

    for feat_x, feat_y in feature_pairs:
        # 2. Compute 1:1 interaction utilizing the predefined conditional lookup pipeline
        print(f"Analyzing interaction : {feat_x} - {feat_y}...")
        res_df = get_local_pairwise_interaction(
                model, instance, background_data, feat_x, n_samples, job_type)
        int_val = res_df.loc[feat_y, 'Interaction']

        interaction_list.append({
            'Feature_X': feat_x,
            'Feature_Y': feat_y,
            'Interaction': int_val
        })
        
    return pd.DataFrame(interaction_list)


#########################################################################
def visualize_feature_contribution(df_res, sample_instance,actual_y, pred_y, scaler=None):
    """
    Plots a horizontal bar chart separating the independent main effect (G1) and interaction effect (G4) for each feature.
    """
    
    if scaler is not None:
        original_instance = scaler.inverse_transform(sample_instance.values.reshape(1, -1))[0]
        original_instance = pd.Series(original_instance, index=sample_instance.index)
    else:
        original_instance = sample_instance    

    plt.close('all')
    
    features = df_res.index
    values = original_instance[features].values  # feature values for the instance
    y_labels = [f"{f}: {v}" for f, v in zip(features, values)]
    
    g1_vals = df_res['Power (G1)'].values
    g4_vals = df_res['Interact (G4)'].values

    plt.figure(figsize=(12, 7))

    for i, feature in enumerate(features):
        g1 = g1_vals[i]
        g4 = g4_vals[i]
        
        # # 1. Plot Main Effect (G1): Always originates from the zero baseline (Teal marker)
        plt.barh(y_labels[i], g1, color='#00bfc4', label='Main effect (G1)' if i==0 else "", alpha=1.0)
        
        ## 2. Plot Interaction (G4): Implements an overlap avoidance layout (Coral marker)
        if np.sign(g1) == np.sign(g4) or g1 == 0:
            # # If signs match, stack horizontally starting from the end of G1 (standard stacking cumulative format)
            plt.barh(y_labels[i], g4, left=g1, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)
        else:
            # If signs conflict, originate from zero and extend in the opposite direction
            # This clearly isolates and highlights the 'diminishing' or suppressive effect described in the paper.
            plt.barh(y_labels[i], g4, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)

# 3. Optimize x-axis boundaries and safety margins automatically
    # Locate extreme ranges among all metrics, absolute combinations, and baseline points
    all_points = np.concatenate([g1_vals, g4_vals, g1_vals + g4_vals, [0]])
    min_x, max_x = np.min(all_points), np.max(all_points)
    
    # Inject a 10% safety buffer based on absolute data range
    x_range = max_x - min_x
    buffer = x_range * 0.1 if x_range > 0 else 0.5
    
    plt.xlim(min_x - buffer, max_x + buffer) # Configure optimized plot boundaries

    plt.axvline(0, color='black', linewidth=1.2)
    plt.title(f"Local Feature Importance\n(Target: Actual={actual_y}, Predict={pred_y})")
    plt.xlabel("Local Importance Value (G1 + G4)")
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

#########################################################################
def visualize_local_pairwise_interaction(df_interact_input, sample_instance, actual_y, pred_y, scaler=None):
    """
    Plots a horizontal bar chart showcasing the 1:1 pairwise interaction strength of all other features with a specific target feature.
    """

    if scaler is not None:
        original_instance = scaler.inverse_transform(sample_instance.values.reshape(1, -1))[0]
        original_instance = pd.Series(original_instance, index=sample_instance.index)
    else:
        original_instance = sample_instance    

    df_interact = df_interact_input.copy()
    target_feature = original_instance.index.difference(df_interact.index)[0]
    df_interact.index = df_interact.index.astype(str) + ": " + original_instance[df_interact.index].astype(str)  # Append feature values to indices 

    plt.close('all')
    
    plt.figure(figsize=(10, 6))
    colors = ['#f8766d' if x > 0 else '#00bfc4' for x in df_interact['Interaction']]
    df_interact['Interaction'].plot(kind='barh', color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Interaction Strength with [{target_feature}: {original_instance[target_feature]}] \n(Target: Actual={actual_y}, Predict={pred_y})")
    plt.xlabel("Interaction Value")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

#########################################################################
import networkx as nx
import matplotlib.cm as cm

def visualize_feature_interaction_graph(importance_data, interaction_df, threshold=1.0):
    """
    Generate a static network graph to visualize feature importance and pairwise feature interactions
    """
    plt.close('all')
    # Set plot style for high-resolution rendering
    plt.style.use('seaborn-v0_8-whitegrid')
    
    G = nx.Graph()
    
    # 1. Populate nodes (Handles both Series and DataFrame configurations)
    if isinstance(importance_data, pd.Series):
        for feat, val in importance_data.items():
            G.add_node(feat, importance=val)
    else:
        col = 'Total' if 'Total' in importance_data.columns else importance_data.columns[0]
        for feat, row in importance_data.iterrows():
            G.add_node(feat, importance=row[col])
            
    for _, row in interaction_df.iterrows():
        u, v, weight = row['Feature_X'], row['Feature_Y'], row['Interaction']
        if abs(weight) > threshold:
            G.add_edge(u, v, weight=weight)

    # 2. Layout Optimization: Adjust spring layout parameter 'k' to expand node spacing
    pos = nx.spring_layout(G, k=1.5, seed=42)
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # 3. Node Visualization: Map sizing and heat coloration relative to overall structural importance
    node_sizes = [max(800, G.nodes[n]['importance'] * 3500) for n in G.nodes]
    node_colors = [G.nodes[n]['importance'] for n in G.nodes]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap=plt.cm.YlOrRd, 
                                   edgecolors='gray', linewidths=0.5, ax=ax)
    
    # Node name and value annotations
    node_labels = {n: f"{n}\n({G.nodes[n]['importance']:.2f})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, 
                           font_weight='bold', font_family='sans-serif', ax=ax)

    # 4. Edge Visualization: Map structural weights into relative line thickness
    edges = G.edges(data=True)
    if len(edges) > 0:
        # Relative thickness mapping: Scale line widths strictly based on (absolute weight - threshold)
        # Maximizes visual contrast by drawing lines near the threshold significantly thinner
        edge_widths = [((abs(d['weight']) - threshold) * 8) + 1 for u, v, d in edges]
        edge_colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges] # Muted gray / orang
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                               alpha=0.5, ax=ax)
        
        # Edge numeric labels: Wrapped in white bounding boxes to guarantee visibility during overlap regimes
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                    font_color='#0000FF', font_weight='bold',
                                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # Plot adjustment
    plt.title(f"Case-Based Feature Interaction Graph (Threshold > {threshold})", 
              fontsize=16, fontweight='bold', pad=30)
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
    cbar.set_label('Feature Importance', rotation=270, labelpad=15)
    
    plt.axis('off')
    plt.margins(0.2)
    plt.tight_layout()
    plt.show()

##########################################################################
class visualize_draggable_interaction_graph:
    """
    Interactive visualization of feature importance and pairwise interactions using a draggable network graph
    """
    def __init__(self, importance_data, interaction_df, threshold=0.0):
        #self.G = G
        #self.pos = pos
        self.importance_data = importance_data
        self.interaction_df = interaction_df
        self.threshold = threshold
        self.selected_node = None
        
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # 1. Instantiate blank graph object
        self.G = nx.Graph()

        # 2. Append Nodes: Assign feature identifier keys and respective importance values
        # Automatically handles variations in input formatting (Series vs DataFrame formats)
        if isinstance(self.importance_data, pd.Series):
            for feat, val in self.importance_data.items():
                self.G.add_node(feat, importance=val)
        else:
            # Query the 'Total' mapping column when using a DataFrame format
            col = 'Total' if 'Total' in self.importance_data.columns else self.importance_data.columns[0]
            for feat, row in self.importance_data.iterrows():
                self.G.add_node(feat, importance=row[col])

        # 3. Append Edges: Establish connectivity weighted by structural synergy strength
        # Restricts edge creation to pairwise relationships exceeding the user-specified threshold value.
        threshold = 1.0 
        for _, row in self.interaction_df.iterrows():
            u, v, weight = row['Feature_X'], row['Feature_Y'], row['Interaction']
            if abs(weight) > self.threshold:
                self.G.add_edge(u, v, weight=weight)

        self.pos = nx.spring_layout(self.G, k=1.5) # initial layout 

        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        
        # 1. Retrieve node metrics
        importances = [self.G.nodes[n]['importance'] for n in self.G.nodes]
        max_imp = max(importances) if max(importances) > 0 else 1
        
        # 2. Optimize node scaling boundaries: Enforce non-zero lower limits and clamp maximum sizes
        # Normalize structural importance indicators to a strict 0~1 domain prior to mapping size vectors
        node_sizes = [max(1000, (self.G.nodes[n]['importance'] / max_imp) * 7000) for n in self.G.nodes]
        node_colors = importances          # Colormap handles internal variable normalization automatically
        
        # 3. Render Graph Node
        nodes = nx.draw_networkx_nodes(self.G, self.pos, node_size=node_sizes, 
                                       node_color=node_colors, cmap=plt.cm.YlOrRd, 
                                       edgecolors='gray', linewidths=2, ax=self.ax)
        
        # 4. Render Layout Labels
        node_labels = {n: f"{n}\n({self.G.nodes[n]['importance']:.2f})" for n in self.G.nodes}
        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=10, font_weight='bold', ax=self.ax)
        
        # 5. Render Graph Edges: Restrict execution to active paired configurations
        edges = self.G.edges(data=True)
        if len(edges) > 0:
            # Map edge thickness proportionally
            weights = [abs(d['weight']) for u, v, d in edges]
            max_w = max(weights) if max(weights) > 0 else 1
            widths = [(w / max_w) * 10 + 1 for w in weights]
            
            colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges]
            nx.draw_networkx_edges(self.G, self.pos, width=widths, edge_color=colors, alpha=0.4, ax=self.ax)
            
            # Optional: Render edge value descriptors)
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8)
        
        self.ax.set_title(f"Feature Interaction Graph (Threshold > {self.threshold})")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        for node, (x, y) in self.pos.items():
            if np.hypot(x - event.xdata, y - event.ydata) < 0.1: # Click detection radius
                self.selected_node = node
                break

    def on_release(self, event):
        if self.selected_node:
            print(f"Final Position '{self.selected_node}': ({self.pos[self.selected_node][0]:.3f}, {self.pos[self.selected_node][1]:.3f})")
        self.selected_node = None

    def on_motion(self, event):
        if self.selected_node is not None and event.inaxes == self.ax:
            self.pos[self.selected_node] = (event.xdata, event.ydata)
            self.update_plot()


