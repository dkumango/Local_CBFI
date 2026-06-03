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
    # 1. 상관계수 기반 주변 변수 선택
    corr_matrix = background_data.corr()
    relevant_features = corr_matrix[target_feature].abs().sort_values(ascending=False)
    cond_features = relevant_features.index[1:3].tolist()
    
    # 2. KNN 학습
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(background_data[cond_features])
    
    query_instance = pd.DataFrame([instance[cond_features]])
    
    dist, indices = nn.kneighbors(query_instance)
    
    return background_data.iloc[indices[0]]

#########################################################################################
def explain_local_cbfi_classification_conditional(model, instance, background_data, target_feature, n_samples=100):
    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    
    # 조건부 샘플링 데이터 확보
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature)
    
    other_features = [f for f in instance.index if f != target_feature]
    counts = {'G1': 0, 'G2': 0, 'G3': 0, 'G4': 0}

    seed=0
    for _ in range(n_samples):
        # 전체가 아닌 조건부 풀에서 샘플링 (핵심 수정 사항)
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
        
        # 논문 그룹화 로직 적용 
        if is_fx_correct and not is_fx_minus_correct: counts['G1'] += 1
        elif not is_fx_correct and is_fx_minus_correct: counts['G2'] += 1
        elif is_fx_correct and is_fx_minus_correct: counts['G3'] += 1
        elif not is_fx_correct and not is_fx_minus_correct: counts['G4'] += 1

    ratios = {k: v / n_samples for k, v in counts.items()}
    return ratios, ratios['G1'] + ratios['G4']

##############################################################################
def explain_local_cbfi_regression_conditional(model, instance, background_data, target_feature, actual, pred, n_samples=100):
    """
    논문 Figure 7의 의사결정 트리 로직을 완벽히 반영하여 G1, G2, G3, G4를 모두 반환합니다.
    """
    pred = model.predict(pd.DataFrame([instance]))[0]
    diff_u = abs(pred - actual)

    # 조건부 샘플링 (이전 단계에서 정의한 _get_conditional_samples 함수 사용)
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature)
    other_features = [f for f in instance.index if f != target_feature]
    
    # 각 그룹별 오차 감소량(Contribution) 합계
    sums = {'G1': 0.0, 'G2': 0.0, 'G3': 0.0, 'G4': 0.0}
    eps = 1e-9 # 부동소수점 비교를 위한 미세값

    seed=0
    for _ in range(n_samples):
        random_sample = conditional_pool.sample(1, random_state=seed).iloc[0]
        seed += 1

        # DS(Fx): Fx 고정, 나머지 치환
        ds_fx = instance.copy()
        for f in other_features: ds_fx[f] = random_sample[f]
            
        # DS(Fx-): Fx 치환, 나머지 고정
        ds_fx_minus = instance.copy()
        ds_fx_minus[target_feature] = random_sample[target_feature]
        
        pred_x = model.predict(pd.DataFrame([ds_fx]))[0]
        pred_minus = model.predict(pd.DataFrame([ds_fx_minus]))[0]
        
        diff_x = abs(pred_x - actual)
        diff_minus = abs(pred_minus - actual)
        
        # 기여도 정의: 기준 오차(diff_minus)에서 최종 오차(diff_u)를 뺀 값 (양수면 오차 감소)
        contribution = diff_minus - diff_u
        
        # --- 논문 Figure 7 의사결정 로직 적용  ---
        
        # 1. G3 (Common Contribution): 오차 변화가 없는 경우 [cite: 373-375]
        if abs(diff_minus - diff_u) < eps:
            sums['G3'] += contribution # 수식상 0에 수렴함 [cite: 399]
            
        # 2. Positive 상황: Fx(-)보다 전체(U)의 오차가 작을 때 [cite: 363]
        elif diff_minus > diff_u:
            if diff_x > diff_u:
                sums['G4'] += contribution # Positive Interaction [cite: 366, 369]
            else:
                sums['G1'] += contribution # Positive Power [cite: 372]
                
        # 3. Negative 상황: 전체(U)의 오차가 Fx(-)보다 클 때 [cite: 376]
        elif diff_u > diff_minus:
            if diff_x > diff_u:
                sums['G1'] += contribution # Negative Power [cite: 379]
            else:
                sums['G4'] += contribution # Negative Interaction [cite: 382, 385]
                
    # 결과 정규화 (n_samples로 나눔)
    ratios = {
        'G1': sums['G1'] / n_samples,
        'G2': 0.0, # 회귀 모델에서는 정의되지 않음 
        'G3': sums['G3'] / n_samples,
        'G4': sums['G4'] / n_samples
    }
    
    # 중요도 합산 (식 21): Imp = FP + Int [cite: 401]
    importance = ratios['G1'] + ratios['G4']
    
    return ratios, importance

########################################################################
def get_cbfi_table(model, instance, background_data, actual=None, pred=None,n_samples=100, job_type='classification'):

    results = []
    features = instance.index
    #target_label = model.predict(pd.DataFrame([instance]))[0]

    print("Calculating Local CBFI for all features...")
    for f in features:
        if job_type == 'classification':
            ratios, _ = explain_local_cbfi_classification_conditional(model, instance, background_data, f, n_samples)
        else:   # regression
            ratios, _ = explain_local_cbfi_regression_conditional(model, instance, background_data, f, actual, pred, n_samples)

        results.append({
            'Feature': f,
            'Power (G1)': ratios['G1'],
            'Others (G2)': ratios['G2'],
            'Common (G3)': ratios['G3'],
            'Interact (G4)': ratios['G4']
        })
    
    df_cbfi_table = pd.DataFrame(results).set_index('Feature')
    # 중요도 합계(G1+G4) 순으로 정렬 
    df_cbfi_table['Total'] = df_cbfi_table['Power (G1)'] + df_cbfi_table['Interact (G4)']
    df_cbfi_table = df_cbfi_table.sort_values(by='Total', ascending=True)

    return df_cbfi_table


## 상호작용 분석
def _local_pairwise_interaction_regression(model, instance, background_data, feat_x, 
                                            n_neighbors=50, 
                                            n_samples=500, 
                                            random_state=42):
    """
    회귀 모델에서 feat_x 와 다른 특징 간의 1:1 상호작용 계산 테이블 
    """
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_y = model.predict(instance_df)[0]

    interaction_results = []

    # 1. 공통 함수: 고정된 샘플 풀을 사용하여 오차(diff) 계산
    def get_diff_vectorized(fixed_feats, sampled_pool):
        diffs = []
        for i in range(n_samples):
            random_sample = sampled_pool.iloc[i]
            temp_ds = instance.copy()
            # 고정된 특징 외에는 샘플 데이터에서 가져옴
            for f in features:
                if f not in fixed_feats:
                    temp_ds[f] = random_sample[f]
            
            pred = model.predict(pd.DataFrame([temp_ds]))[0]
            diffs.append(abs(pred - target_y))
        return np.array(diffs)

    # 특징 y별로 루프를 돌며 상호작용 계산
    for feat_y in features:
        if feat_x == feat_y: continue
        
        # 2. {Fx, Fy} 기준의 국소적 이웃 추출 (분류 모델과 동일 로직으로 일관성 유지)
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        
        # 3. 재현성을 위해 이웃 풀 내에서 n_samples만큼 미리 복원 추출
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        # 4. 각 시나리오별 오차(diff) 계산
        diff_x = get_diff_vectorized([feat_x], sampled_pool)
        diff_y = get_diff_vectorized([feat_y], sampled_pool)
        diff_xy = get_diff_vectorized([feat_x, feat_y], sampled_pool)
        
        # 5. 논문 식 (22) 적용: 오차 감소량의 평균
        # Interaction = (E[diff_x - diff_xy] + E[diff_y - diff_xy]) / 2
        int_val = (np.mean(diff_x - diff_xy) + np.mean(diff_y - diff_xy)) / 2
        
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': int_val})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False) 
 


def _local_pairwise_interaction_classification(model, instance, background_data, feat_x, 
                                               n_neighbors=50, # 국소성 유지 (작게)
                                               n_samples=500,  # 통계적 안정성 (크게)
                                               random_state=42):
    """
    분류 모델에서 feat_x 와 다른 특징 간의 조건부 1:1 상호작용 분석 (논문 Section 2.3 및 식 14 기반)
    """
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    
    interaction_results = []
    
    for feat_y in features:
        if feat_x == feat_y: continue
        
        # 1. 조건부 샘플링 풀 생성
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        # 1. 고정된 크기의 이웃 풀 생성 (국소성 확보)
        nn = NearestNeighbors(n_neighbors=n_neighbors) 
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        
        # 2. 작은 이웃 풀 내에서 500번 복원 추출 (재현성 및 안정성 확보)
        # n_samples가 pool 크기보다 크므로 replace=True 필수
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        g4_count = 0
        
        for i in range(n_samples):
            random_sample = sampled_pool.iloc[i] # 고정된 순서로 샘플 접근
            
            # 데이터셋 구성 로직 (이전과 동일)
            ds_xy, ds_x, ds_y = instance.copy(), instance.copy(), instance.copy()
            
            for f in features:
                if f not in [feat_x, feat_y]:
                    ds_xy[f] = random_sample[f]
                if f != feat_x:
                    ds_x[f] = random_sample[f]
                if f != feat_y:
                    ds_y[f] = random_sample[f]
            
            # 예측 수행
            pred_xy = model.predict(pd.DataFrame([ds_xy]))[0]
            pred_x = model.predict(pd.DataFrame([ds_x]))[0]
            pred_y = model.predict(pd.DataFrame([ds_y]))[0]
            
            if (pred_xy == target_label) and (pred_x != target_label) and (pred_y != target_label):
                g4_count += 1
        
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': g4_count / n_samples})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False)


def get_local_pairwise_interaction(model, instance, background_data, feat_x, n_samples=100, job_type='classification'):
    if job_type == 'classification':
        return _local_pairwise_interaction_classification(model, instance, background_data, feat_x, n_samples)
    else:   # regression
        return _local_pairwise_interaction_regression(model, instance, background_data, feat_x,  n_samples=n_samples)

########################################################################
from itertools import combinations

def generate_all_interactions(model, instance, background_data, n_samples=100, job_type='classification'):
    """
    모든 특징 쌍 사이의 상호작용을 계산하여 그래프용 DataFrame 생성
    """
    features = instance.index.tolist()
    interaction_list = []
    
    # 1. 모든 특징 쌍(Combination) 생성 [cite: 271, 593]
    feature_pairs = list(combinations(features, 2))
    
    print(f"Total {len(feature_pairs)} pairs to analyze...")

    for feat_x, feat_y in feature_pairs:
        # 2. 1:1 상호작용 계산 (이전 단계에서 정의한 조건부 함수 활용)
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
        
        # 1. Main Effect (G1) 그리기: 항상 0에서 시작 (Teal)
        plt.barh(y_labels[i], g1, color='#00bfc4', label='Main effect (G1)' if i==0 else "", alpha=1.0)
        
        # 2. Interaction (G4) 그리기: 겹침 방지 로직 (Coral)
        if np.sign(g1) == np.sign(g4) or g1 == 0:
            # 부호가 같으면 G1 끝에서 시작 (정상 누적)
            plt.barh(y_labels[i], g4, left=g1, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)
        else:
            # 부호가 다르면 0에서 시작하여 반대 방향으로 뻗음 (겹침 방지 및 분리 설명)
            # 논문의 'diminishing' 효과를 가장 명확히 보여주는 방식입니다.
            plt.barh(y_labels[i], g4, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)

# 2. x축 범위(여백) 자동 최적화 로직 (핵심 수정 사항)
    # 모든 성분 및 합산값 중 최소/최대 탐색
    all_points = np.concatenate([g1_vals, g4_vals, g1_vals + g4_vals, [0]])
    min_x, max_x = np.min(all_points), np.max(all_points)
    
    # 데이터 범위의 10% 정도만 여백으로 추가
    x_range = max_x - min_x
    buffer = x_range * 0.1 if x_range > 0 else 0.5
    
    plt.xlim(min_x - buffer, max_x + buffer) # 최적화된 범위 설정

    plt.axvline(0, color='black', linewidth=1.2)
    plt.title(f"Local Feature Importance\n(Target: Actual={actual_y}, Predict={pred_y})")
    plt.xlabel("Local Importance Value (G1 + G4)")
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

#########################################################################
def visualize_local_pairwise_interaction(df_interact_input, sample_instance, actual_y, pred_y, scaler=None):
    
    if scaler is not None:
        original_instance = scaler.inverse_transform(sample_instance.values.reshape(1, -1))[0]
        original_instance = pd.Series(original_instance, index=sample_instance.index)
    else:
        original_instance = sample_instance    

    df_interact = df_interact_input.copy()
    target_feature = original_instance.index.difference(df_interact.index)[0]
    df_interact.index = df_interact.index.astype(str) + ": " + original_instance[df_interact.index].astype(str)  # 인덱스에 feature value 추가 

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
    plt.close('all')
    # 고해상도 설정을 위해 plt 스타일 지정
    plt.style.use('seaborn-v0_8-whitegrid')
    
    G = nx.Graph()
    
    # 1. 데이터 입력 (Series/DataFrame 대응)
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

    # 2. 레이아웃 최적화: 노드 간 간격을 넓히기 위해 k값 조절 [cite: 594]
    pos = nx.spring_layout(G, k=1.5, seed=42)
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # 3. 노드 시각화: 중요도에 따른 크기 및 색상 (Heat color) [cite: 588-590]
    node_sizes = [max(800, G.nodes[n]['importance'] * 3500) for n in G.nodes]
    node_colors = [G.nodes[n]['importance'] for n in G.nodes]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap=plt.cm.YlOrRd, 
                                   edgecolors='gray', linewidths=0.5, ax=ax)
    
    # 노드 이름/수치 라벨
    node_labels = {n: f"{n}\n({G.nodes[n]['importance']:.2f})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, 
                           font_weight='bold', font_family='sans-serif', ax=ax)

    # 4. 에지 시각화: 임계값 대비 상대적 두께 적용 [cite: 591-592]
    edges = G.edges(data=True)
    if len(edges) > 0:
        # 상대적 두께 계산 로직: (절대값 - 임계값)에 비례하여 굵기 결정
        # 임계값에 가까운 선은 가늘게, 멀어질수록 굵게 표현하여 시각적 대비 극대화
        edge_widths = [((abs(d['weight']) - threshold) * 8) + 1 for u, v, d in edges]
        edge_colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges] # 부드러운 회색/오렌지
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                               alpha=0.5, ax=ax)
        
        # 에지 수치 라벨: 배경색(bbox)을 넣어 글자가 겹쳐도 보이게 처리
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                    font_color='#0000FF', font_weight='bold',
                                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # 마무리 설정
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

        # 1. 빈 그래프 객체 생성
        self.G = nx.Graph()

        # 2. 노드(Node) 추가: 특징 이름과 중요도 수치 부여 [cite: 587-588]
        # 데이터가 Series인지 DataFrame인지에 따라 자동으로 처리합니다.
        if isinstance(self.importance_data, pd.Series):
            for feat, val in self.importance_data.items():
                self.G.add_node(feat, importance=val)
        else:
            # DataFrame인 경우 'Total' 컬럼 사용 [cite: 194, 401]
            col = 'Total' if 'Total' in self.importance_data.columns else self.importance_data.columns[0]
            for feat, row in self.importance_data.iterrows():
                self.G.add_node(feat, importance=row[col])

        # 3. 에지(Edge) 추가: 상호작용 강도 부여 [cite: 587, 591]
        # threshold보다 큰 상호작용만 그래프의 선으로 연결합니다[cite: 594].
        threshold = 1.0 
        for _, row in self.interaction_df.iterrows():
            u, v, weight = row['Feature_X'], row['Feature_Y'], row['Interaction']
            if abs(weight) > self.threshold:
                self.G.add_edge(u, v, weight=weight)

        self.pos = nx.spring_layout(self.G, k=1.5) # 초기 배치

        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        
        # 1. 노드 수치 가져오기
        importances = [self.G.nodes[n]['importance'] for n in self.G.nodes]
        max_imp = max(importances) if max(importances) > 0 else 1
        
        # 2. 노드 크기 최적화: 중요도에 비례하되 최대 크기를 제한 (Scaling)
        # 중요도를 최대값으로 나누어 0~1 사이로 만든 뒤 크기 부여
        node_sizes = [max(1000, (self.G.nodes[n]['importance'] / max_imp) * 7000) for n in self.G.nodes]
        node_colors = importances # 컬러맵은 자동 스케일링을 지원함
        
        # 3. 노드 그리기
        nodes = nx.draw_networkx_nodes(self.G, self.pos, node_size=node_sizes, 
                                       node_color=node_colors, cmap=plt.cm.YlOrRd, 
                                       edgecolors='gray', linewidths=2, ax=self.ax)
        
        # 4. 레이블 그리기
        node_labels = {n: f"{n}\n({self.G.nodes[n]['importance']:.2f})" for n in self.G.nodes}
        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=10, font_weight='bold', ax=self.ax)
        
        # 5. 에지(Edge) 그리기: 상호작용이 존재하는 경우만
        edges = self.G.edges(data=True)
        if len(edges) > 0:
            # 에지 두께도 상대적으로 조절
            weights = [abs(d['weight']) for u, v, d in edges]
            max_w = max(weights) if max(weights) > 0 else 1
            widths = [(w / max_w) * 10 + 1 for w in weights]
            
            colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges]
            nx.draw_networkx_edges(self.G, self.pos, width=widths, edge_color=colors, alpha=0.4, ax=self.ax)
            
            # 에지 레이블 (선택 사항)
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8)
        
        self.ax.set_title(f"Feature Interaction Graph (Threshold > {self.threshold})")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        for node, (x, y) in self.pos.items():
            if np.hypot(x - event.xdata, y - event.ydata) < 0.1: # 클릭 반경
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


