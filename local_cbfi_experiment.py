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
# CLASSIFICATION
##############################################################################

def load_and_preprocess_wine_data(random_seed=42):
    
    df = pd.read_csv("dataset/winequality-red.csv") 
    df.drop_duplicates(inplace=True)     # 중복제거
    
    # 2. 분석의 편의성을 위한 처리
    X = df.drop('quality', axis=1)
    y = df['quality']

    # 3. 학습/테스트 데이터 분리 (충분한 샘플 수 확보를 위해 8:2 분할)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # 4. 특징 스케일링 (StandardScaler)
    # CBFI의 _get_conditional_samples() 내 KNN 연산을 위해 필수적인 단계입니다.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # 인덱스 초기화 (추후 iloc 접근 편의성)
    X_train_scaled.reset_index(drop=True, inplace=True)
    X_test_scaled.reset_index(drop=True, inplace=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 실행 및 데이터 확인
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_wine_data()
print(f"Dataset Loaded: {len(X_train)} train samples, {len(X_test)} test samples")

# RF model build
model_wine = RandomForestClassifier(n_estimators=50,  random_state=42)
model_wine.fit(X_train, y_train)
# model_wine.score(X_test, y_test)
print(f"Random Forest Test Accuracy: {model_wine.score(X_test, y_test):.4f}")

#############
## CBFI #####
#############

pick = 8  # 0부터 시작하는 인덱스, 예: 18번 샘플 (quality:7)
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
interaction_df_1 = pd.read_csv("interaction_df_1.csv")  # 이미 계산된 상호작용 데이터 로드
#interaction_df_1 = CBFI.generate_all_interactions(model_wine, sample_instance, X_train, n_samples=100, job_type='classification')
## CBFI.visualize_feature_interaction_graph(importance_df_1, interaction_df_1, threshold=0.01) # static
#interaction_df_1.to_csv("interaction_df_1.csv", index=False)

CBFI.visualize_draggable_interaction_graph(importance_df_1, interaction_df_1, threshold=0.001) # interactive

#############
## SHAP #####
#############

import shap

# 1. SHAP Explainer 초기화 (트리 모델 전용)
explainer_shap = shap.TreeExplainer(model_wine)

# 2. 테스트 인스턴스에 대한 SHAP Value 계산
# 멀티 클래스이므로 shap_values는 [클래스 개수, 샘플 수, 특징 수] 형태의 리스트입니다.
shap_values = explainer_shap.shap_values(sample_instance)

# 3. 특정 샘플(test_idx)의 예측 클래스에 대한 결과 시각화
# 예측된 클래스의 인덱스를 찾습니다 (예: 예측이 6점이면 클래스 리스트 중 해당 위치)
pred_class_idx = list(model_wine.classes_).index(predicted_1)

print(f"SHAP Analysis for Predicted Class: {predicted_1}")
#shap.initjs() # 주피터 노트북 환경 시 필요
shap.force_plot(
    explainer_shap.expected_value[pred_class_idx], 
    shap_values[:,pred_class_idx],          # feature importance for the predicted class
    sample_instance,
    matplotlib=True
)
# SHAP feature importance
SHAP_imp = pd.Series(shap_values[:, pred_class_idx], index=X_train.columns)
CBFI_imp = df_cbfi_table_1['Total']
CBFI_imp = CBFI_imp.reindex(sample_instance.index)  # SHAP과 동일한 feature 순서로 정렬
SHAP_imp.corr(CBFI_imp)
CBFI_imp.corr(np.abs(SHAP_imp))

# feature interaction
# 주의: 일반 SHAP 값보다 계산 시간이 훨씬(기하급수적으로) 오래 걸립니다.
interaction_values = explainer_shap.shap_interaction_values(X_test.iloc[pick:pick+1])

# 결과의 shape: (샘플 수, 특징 수, 특징 수)
# 18번 샘플의 11x11 상호작용 행렬 확인
sample_interaction = interaction_values[0][10,:,pred_class_idx]
# => 이 결과 시각화 해야됨

# G4 와 SHAP dml corrrelation 계산
G4 = df_interact_1.iloc[:,0].to_list()   # feature 순서를 원랟대로 해야되
SHAP_interact = sample_interaction[:10].tolist()  # 대각선 제외한 상호작용 값 (자기 자신과의 상호작용은 제외)

correlation = np.corrcoef(G4, SHAP_interact)[0, 1]
print(f"Correlation between CBFI G4 and SHAP Interaction Values: {correlation:.4f}")

###########
## LIME
############

from lime import lime_tabular

# 1. LIME Explainer 초기화
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=model_wine.classes_.astype(str).tolist(),
    mode='classification'
)

# 2. 테스트 인스턴스 설명 생성
# num_features는 상위 몇 개의 특징을 보여줄지 결정합니다.

class_idx = list(model_wine.classes_).index(predicted_1)
exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[pick].values, 
    predict_fn=model_wine.predict_proba,
    num_features=X_test.shape[1],  # 모든 특징을 보여줌
    labels=[class_idx] # 예측된 클래스에 대한 설명 요청
)

# 3. 결과 시각화
#exp.show_in_notebook(show_table=True)
# 또는 리스트로 확인

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
    특징 제거에 따른 예측 확률 하락폭(Fidelity) 측정 (인덱스 오류 수정 버전)
    """
    import pandas as pd
    import numpy as np

    feature_names = instance.index
    instance_df = pd.DataFrame([instance])
    
    # 1. 모델의 예측 라벨 확인
    target_label = model.predict(instance_df)[0]
    
    # 2. 라벨 값(예: 7)을 확률 배열의 인덱스(예: 4)로 변환 (핵심 수정 사항)
    target_idx = list(model.classes_).index(target_label)
    
    # 3. 초기 확률값 추출
    initial_proba = model.predict_proba(instance_df)[0][target_idx]
    
    reference_values = background_data.median()
    results = {}
    
    for method_name, importance_values in importance_dict.items():
        # 중요도 순으로 특징 정렬
        sorted_features = pd.Series(importance_values, index=feature_names).abs().sort_values(ascending=False).index
        
        probas = [initial_proba]
        temp_instance = instance.copy()
        
        # 상위 n_steps개의 특징을 순차적으로 제거
        for i in range(min(n_steps, len(sorted_features))):
            feat_to_remove = sorted_features[i]
            temp_instance[feat_to_remove] = reference_values[feat_to_remove]
            
            # 매번 변환된 인덱스(target_idx)를 사용하여 확률 추출
            new_proba = model.predict_proba(pd.DataFrame([temp_instance]))[0][target_idx]
            probas.append(new_proba)
            
        results[method_name] = probas
        
    return results

# 1. 중요도 데이터 준비 (교수님께서 공유해주신 수치)
importance_dict = {
    'SHAP': SHAP_imp, # 원본 SHAP (함수 내에서 abs 처리)
    'LIME': LIME_imp, 
    'CBFI': CBFI_imp  # G1 + G4 합산
}

# 2. 실험 실행
fidelity_results = perform_fidelity_test(model_wine, X_test.iloc[pick], importance_dict, X_train)

# 3. 시각화
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
    상호작용 변수 쌍의 동시 제거에 따른 시너지 효과 측정
    """
    feature_names = instance.index
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    initial_proba = model.predict_proba(pd.DataFrame([instance]))[0][target_idx]
    
    # 대체값(중앙값) 설정
    ref_vals = background_data.median()
    
    # 1. 테스트할 변수 쌍 선정
    # CBFI: G4(Interaction)가 가장 높은 상위 2개 변수
    top_g4_pair = cbfi_df['Interact (G4)'].sort_values(ascending=False).index[:2].tolist()
    
    # SHAP: 절대값 기여도가 가장 높은 상위 2개 변수
    top_shap_pair = shap_df.abs().sort_values(ascending=False).index[:2].tolist()
    
    pairs = {'CBFI (G4 Pair)': top_g4_pair, 'SHAP (Top Pair)': top_shap_pair}
    results = []

    for label, pair in pairs.items():
        fx, fy = pair[0], pair[1]
        
        # 시나리오별 샘플 생성
        inst_x = instance.copy(); inst_x[fx] = ref_vals[fx] # Fx 제거
        inst_y = instance.copy(); inst_y[fy] = ref_vals[fy] # Fy 제거
        inst_xy = instance.copy(); inst_xy[fx] = ref_vals[fx]; inst_xy[fy] = ref_vals[fy] # 둘 다 제거
        
        # 확률 측정
        p_x = model.predict_proba(pd.DataFrame([inst_x]))[0][target_idx]
        p_y = model.predict_proba(pd.DataFrame([inst_y]))[0][target_idx]
        p_xy = model.predict_proba(pd.DataFrame([inst_xy]))[0][target_idx]
        
        # 하락폭(Drop) 계산
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

# --- 실행 예시 ---
# 교수님의 데이터프레임 변수명에 맞춰 실행하십시오.
# cbfi_df: G1~G4가 포함된 테이블, shap_imp: SHAP 수치 시리즈
synergy_results = perform_synergy_test(model_wine, X_test.iloc[pick], df_cbfi_table_1, SHAP_imp, X_train)
print(synergy_results)

##########################
# Drop(alcohol)과 Drop(total sulfur dioxide)의 개별 값을 확인
##########################
def check_individual_drops(model, instance, background_data):
    """
    특정 특징들을 각각 개별적으로 제거했을 때의 확률 하락폭을 측정
    """
    feature_names = instance.index
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    initial_proba = model.predict_proba(pd.DataFrame([instance]))[0][target_idx]

    # 대체 기준값 (훈련 데이터의 중앙값)
    ref_vals = background_data.median()

    features_to_check = ['alcohol', 'total sulfur dioxide']
    results = {}

    for feat in features_to_check:
        temp_instance = instance.copy()
        temp_instance[feat] = ref_vals[feat]
        
        # 특징 제거 후 확률 계산
        new_proba = model.predict_proba(pd.DataFrame([temp_instance]))[0][target_idx]
        drop_val = initial_proba - new_proba
        
        results[feat] = {
            'Original_Proba': initial_proba,
            'New_Proba': new_proba,
            'Drop': drop_val
        }

    return pd.DataFrame(results).T

# 실행 (pick=18 샘플 기준)
individual_drops = check_individual_drops(model_wine, X_test.iloc[pick], X_train)
print(individual_drops)

#################################################################
# *알코올 수치를 변화시키면서 아황산염 유무에 따른 확률 변화 곡선(Partial Dependence 스타일)**을 그려보는 실험
#################################################################
def plot_interaction_sensitivity(model, instance, background_data, target_feat='alcohol', interact_feat='total sulfur dioxide'):
    """
    주요 변수(alcohol) 변화에 따른 인터랙션 변수(SO2)의 조절 효과 시각화
    """
    # 1. 준비 작업
    target_label = model.predict(pd.DataFrame([instance]))[0]
    target_idx = list(model.classes_).index(target_label)
    ref_val = background_data[interact_feat].median()
    
    # 2. 알코올 변화 범위 설정 (데이터셋의 최소~최대 범위)
    x_range = np.linspace(background_data[target_feat].min(), background_data[target_feat].max(), 100)
    
    proba_with_so2 = []
    proba_without_so2 = []
    
    for val in x_range:
        # 시나리오 A: 원래 SO2 상태 유지
        inst_a = instance.copy()
        inst_a[target_feat] = val
        p_a = model.predict_proba(pd.DataFrame([inst_a]))[0][target_idx]
        proba_with_so2.append(p_a)
        
        # 시나리오 B: SO2 제거 (중앙값으로 대체)
        inst_b = instance.copy()
        inst_b[target_feat] = val
        inst_b[interact_feat] = ref_val
        p_b = model.predict_proba(pd.DataFrame([inst_b]))[0][target_idx]
        proba_without_so2.append(p_b)
        
    # 3. 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, proba_with_so2, label=f'With {interact_feat} (Original)', color='teal', linewidth=2)
    plt.plot(x_range, proba_without_so2, label=f'Without {interact_feat} (Median)', color='salmon', linestyle='--', linewidth=2)
    
    # 현재 인스턴스의 실제 알코올 위치 표시
    plt.axvline(x=instance[target_feat], color='gray', linestyle=':', label=f'Current {target_feat} ({instance[target_feat]})')
    
    plt.title(f"Interaction Effect: How '{interact_feat}' regulates '{target_feat}'", fontsize=14)
    plt.xlabel(f"{target_feat} Content", fontsize=12)
    plt.ylabel(f"Probability of Class {target_label}", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 실행
plot_interaction_sensitivity(model_wine, X_test.iloc[pick], X_train)


##############################################################################
# REGRESSION
##############################################################################
def load_and_preprocess_insurance_data(random_seed=42):

    df = pd.read_csv("dataset/Medical_Cost.csv")
    
    # 2. 범주형 변수 인코딩
    le = LabelEncoder()
    # smoker: yes -> 1, no -> 0 (상호작용 분석 시 매우 중요)
    df['smoker'] = le.fit_transform(df['smoker'])
    # sex: male, female 변환
    df['sex'] = le.fit_transform(df['sex'])
    # region: 4개 지역 변환
    df['region'] = le.fit_transform(df['region'])
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    # 4. 스케일링 (KNN 조건부 샘플링의 안정성 확보)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 실행
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
interaction_df_2 = pd.read_csv("interaction_df_2_insurance.csv")  # 이미 계산된 상호작용 데이터 로드
#interaction_df_2 = CBFI.generate_all_interactions(model_insurance, sample_instance, X_train, n_samples=100, job_type='regression')
#interaction_df_2.to_csv("interaction_df_2_insurance.csv", index=False)

CBFI.visualize_draggable_interaction_graph(importance_df_2, interaction_df_2, threshold=100) # interactive


#############
## SHAP #####
#############

import shap

# 1. SHAP Explainer 초기화 (회귀 모델용)
explainer_shap = shap.TreeExplainer(model_insurance)

# 2. 테스트 인스턴스에 대한 SHAP Value 계산
# 회귀 모델의 shap_values는 [샘플 수, 특징 수] 형태의 단일 array입니다.
shap_values = explainer_shap.shap_values(sample_instance)

# 3. 특정 샘플(pick) 시각화 (회귀는 클래스 선택 과정이 필요 없음)
print(f"SHAP Analysis for Predicted Insurance Charges")

base_val = explainer_shap.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[0]

# 2. Data 값 추출 (Series인지 DataFrame인지에 따라 대응)
# 만약 sample_instance가 Series라면 바로 .values를, DataFrame이라면 .iloc[0].values를 씁니다.
if isinstance(sample_instance, pd.Series):
    actual_data = sample_instance.values
else:
    actual_data = sample_instance.iloc[0].values

# 3. Explanation 객체 생성
exp = shap.Explanation(
    values=shap_values[0],          # SHAP 기여도 배열
    base_values=base_val,           # 모델의 평균 예측값
    data=actual_data,               # 실제 특징 값들
    feature_names=X_train.columns.tolist()
)

# 4. SHAP Feature Importance 추출 및 상관관계 계산
# 회귀에서는 단순히 shap_values[0]이 각 특징의 기여도입니다.
SHAP_imp = pd.Series(shap_values, index=X_train.columns)
CBFI_imp = df_cbfi_table_2['Total']
CBFI_imp = CBFI_imp.reindex(SHAP_imp.index) # 특징 순서 정렬

corr_raw = SHAP_imp.corr(CBFI_imp)
corr_abs = CBFI_imp.corr(np.abs(SHAP_imp))

print(f"Correlation (Raw): {corr_raw:.4f}")
print(f"Correlation (Absolute): {corr_abs:.4f}")

# 5. SHAP Interaction Values 계산
# 결과 shape: (샘플 수, 특징 수, 특징 수)
interaction_values = explainer_shap.shap_interaction_values(sample_instance)
# 18번 샘플의 [특징 수 x 특징 수] 상호작용 행렬 추출
matrix_interaction = interaction_values 

# 6. 특정 변수(예: feat_x)에 대한 SHAP Interaction 값 추출
# 교수님 코드의 feat_x가 index 0번(예: age)이라면 해당 행을 가져옵니다.
feat_x_idx = 4 # 예시로 'smoker' 변수의 인덱스  
shap_interact_row = matrix_interaction[feat_x_idx]

# 7. CBFI G4와 SHAP Interaction 간의 상관관계 계산
# SHAP 행렬의 대각선(Self-interaction)을 제외하고 CBFI의 G4 결과와 매칭합니다.
# df_interact_2은 feat_x와 다른 모든 feat_y 간의 G4 값이 들어있는 테이블이어야 합니다.
G4_values = df_interact_2['Interaction'].tolist()

# SHAP 상호작용 행렬에서 feat_x 행을 가져오되, 자기 자신(대각선)은 제외
mask = np.ones(len(shap_interact_row), dtype=bool)
mask[feat_x_idx] = False
SHAP_interact_filtered = shap_interact_row[mask].tolist()

correlation_g4 = np.corrcoef(G4_values, SHAP_interact_filtered)[0, 1]
print(f"Correlation between CBFI G4 and SHAP Interaction: {correlation_g4:.4f}")
# 결과 :  0.2430


###########
## LIME
############

from lime import lime_tabular

# 1. LIME Explainer 초기화
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    mode='regression'
)

# 2. 테스트 인스턴스 설명 생성
# num_features는 상위 몇 개의 특징을 보여줄지 결정합니다.
exp = explainer_lime.explain_instance(
    data_row=X_test.iloc[pick].values, 
    predict_fn=model_insurance.predict,
    num_features=X_test.shape[1]  # 모든 특징을 보여줌
)

# 3. 결과 시각화
#exp.show_in_notebook(show_table=True)
# 또는 리스트로 확인

print("--- LIME Feature Importance List ---")
print(exp.as_list())

# LIME 회귀 모드에서는 보통 1번 인덱스에 결과가 저장됩니다.
# 만약 에러가 발생한다면 list(exp.local_exp.keys())[0]으로 유연하게 대처 가능합니다.
target_key = list(exp.local_exp.keys())[0]
local_exp = exp.local_exp[target_key]

# 특징 이름 매칭 및 Series 생성
LIME_imp = pd.Series({
    explainer_lime.feature_names[idx]: val for idx, val in local_exp
}, name='importance')

#  특징 순서 정렬 (SHAP, CBFI와 동일하게 정렬)
LIME_imp = LIME_imp.reindex(sample_instance.index)

#  상관관계 계산
# CBFI와의 타당성 검토 (절대값 기준) [cite: 45-46, 59-60]
corr_cbfi_lime = CBFI_imp.corr(np.abs(LIME_imp))
# SHAP과의 일관성 검토 (가산적 모델 간 비교) 
corr_lime_shap = LIME_imp.corr(SHAP_imp)

print(f"Correlation (CBFI vs |LIME|): {corr_cbfi_lime:.4f}")
print(f"Correlation (LIME vs SHAP): {corr_lime_shap:.4f}")

#############################################################################
# Fidelity Test (Regression)
#############################################################################
import matplotlib.pyplot as plt

def perform_fidelity_test_regression(model, instance, importance_dict, background_data, n_steps=5):
    """
    특징 제거에 따른 예측 수치(Regression Value) 변화 측정
    """
    feature_names = instance.index
    instance_df = pd.DataFrame([instance])
    
    # 1. 초기 예측값 추출 (회귀이므로 수치 자체가 결과임)
    initial_pred = model.predict(instance_df)[0]
    
    # 특징 제거 시 대체할 기준값 (훈련 데이터의 중앙값)
    reference_values = background_data.median()
    results = {}
    
    for method_name, importance_values in importance_dict.items():
        # 중요도 순으로 특징 정렬 (절대값 기준)
        sorted_features = pd.Series(importance_values, index=feature_names).abs().sort_values(ascending=False).index
        
        preds = [initial_pred]
        temp_instance = instance.copy()
        
        # 상위 n_steps개의 특징을 순차적으로 제거(중앙값으로 대체)
        for i in range(min(n_steps, len(sorted_features))):
            feat_to_remove = sorted_features[i]
            temp_instance[feat_to_remove] = reference_values[feat_to_remove]
            
            # 특징 제거 후 새로운 예측값 계산
            new_pred = model.predict(pd.DataFrame([temp_instance]))[0]
            preds.append(new_pred)
            
        results[method_name] = preds
        
    return results


# 1. 중요도 데이터 준비 (교수님께서 공유해주신 수치)
importance_dict = {
    'SHAP': SHAP_imp, # 원본 SHAP (함수 내에서 abs 처리)
    'LIME': LIME_imp, 
    'CBFI': CBFI_imp  # G1 + G4 합산
}

# 2. 실험 실행
fidelity_results = perform_fidelity_test_regression(model_insurance, X_test.iloc[pick], importance_dict, X_train)

# 3. 시각화
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
    회귀 모델에서 상호작용 변수 쌍의 동시 제거에 따른 예측값 변화(시너지) 측정
    """
    feature_names = instance.index
    # 1. 초기 예측값 추출 (회귀 결과)
    initial_pred = model.predict(pd.DataFrame([instance]))[0]
    
    # 대체값(중앙값) 설정
    ref_vals = background_data.median()
    
    # 2. 테스트할 변수 쌍 선정
    # CBFI: G4(Interaction)가 가장 높은 상위 2개 변수
    top_g4_pair = cbfi_df['Interact (G4)'].sort_values(ascending=False).index[:2].tolist()
    
    # SHAP: 절대값 기여도가 가장 높은 상위 2개 변수 (회귀 SHAP_imp 기준)
    top_shap_pair = shap_df.abs().sort_values(ascending=False).index[:2].tolist()
    
    pairs = {'CBFI (G4 Pair)': top_g4_pair, 'SHAP (Top Pair)': top_shap_pair}
    results = []

    for label, pair in pairs.items():
        fx, fy = pair[0], pair[1]
        
        # 시나리오별 샘플 생성
        inst_x = instance.copy(); inst_x[fx] = ref_vals[fx] # Fx 제거
        inst_y = instance.copy(); inst_y[fy] = ref_vals[fy] # Fy 제거
        inst_xy = instance.copy(); inst_xy[fx] = ref_vals[fx]; inst_xy[fy] = ref_vals[fy] # 둘 다 제거
        
        # 예측값 측정 (predict_proba 대신 predict 사용)
        pred_x = model.predict(pd.DataFrame([inst_x]))[0]
        pred_y = model.predict(pd.DataFrame([inst_y]))[0]
        pred_xy = model.predict(pd.DataFrame([inst_xy]))[0]
        
        # 변화폭(Drop) 계산: 초기값과 제거 후 값의 차이
        drop_x = initial_pred - pred_x
        drop_y = initial_pred - pred_y
        drop_joint = initial_pred - pred_xy
        
        # 시너지 지수 계산 (LaTeX: $$SI = \frac{Drop_{joint}}{Drop_x + Drop_y}$$)
        synergy_index = drop_joint / (drop_x + drop_y + 1e-9)
        
        results.append({
            'Method': label,
            'Pair': f"{fx} & {fy}",
            'Individual_Sum': drop_x + drop_y,
            'Joint_Drop': drop_joint,
            'Synergy_Index': synergy_index
        })

    return pd.DataFrame(results)

# --- 실행 예시 ---
synergy_results = perform_synergy_test_regression(model_insurance, X_test.iloc[pick], df_cbfi_table_2, SHAP_imp, X_train)
print(synergy_results)
#             Method          Pair  Individual_Sum   Joint_Drop  Synergy_Index
# 0   CBFI (G4 Pair)  smoker & age    -6646.155188 -6646.155188            1.0
# 1  SHAP (Top Pair)  age & smoker    -6646.155188 -6646.155188            1.0



