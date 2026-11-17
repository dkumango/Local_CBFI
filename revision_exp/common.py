"""
Shared data loaders, model factory and helpers for the revision experiments.

Leakage-free by construction: every model is a sklearn Pipeline(StandardScaler -> estimator)
fitted on the training split only; explanations operate on raw (unscaled) feature values.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "dataset")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
MAX_TRAIN = 20000          # cap on training rows (documented in the paper) to keep SVM/TabNet tractable
BACKGROUND_N = 1000        # background rows used by explainers


# --------------------------------------------------------------------------------------
# Datasets  (name -> (X, y, task))
# --------------------------------------------------------------------------------------
def _bool_to_int(df):
    b = df.select_dtypes(include=["bool"]).columns
    df[b] = df[b].astype(int)
    return df


def load_wine():
    df = pd.read_csv(os.path.join(DATA, "winequality-red.csv")).drop_duplicates()
    return df.drop(columns="quality"), df["quality"], "clf"


def load_insurance():
    df = pd.read_csv(os.path.join(DATA, "Medical_Cost.csv"))
    df["sex"] = (df["sex"] == "male").astype(int)
    df["smoker"] = (df["smoker"] == "yes").astype(int)
    df = pd.get_dummies(df, columns=["region"], drop_first=True)
    df = _bool_to_int(df)
    return df.drop(columns="charges"), df["charges"], "reg"


def load_adult():
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income']
    df = pd.read_csv(os.path.join(DATA, "adult.data.csv"), names=cols, skipinitialspace=True)
    df = df.replace("?", np.nan).dropna().drop(columns=["education"])   # education == education-num
    y = (df.pop("income").str.strip() == ">50K").astype(int)
    X = _bool_to_int(pd.get_dummies(df, drop_first=True))
    return X, y, "clf"


def load_bank():
    df = pd.read_csv(os.path.join(DATA, "bank.csv"), sep=None, engine="python", skipinitialspace=True)
    # 'duration' is only known after the call (target leakage per the UCI description) -> removed
    df = df.drop(columns=["duration"])
    df = df.replace("unknown", np.nan).dropna(axis=1, thresh=int(len(df) * 0.8)).dropna()
    y = (df.pop("y") == "yes").astype(int)
    X = _bool_to_int(pd.get_dummies(df, drop_first=True))
    return X, y, "clf"


def load_diabetes():
    df = pd.read_csv(os.path.join(DATA, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")).dropna()
    y = df.pop("Diabetes_binary").astype(int)
    return df.astype(float), y, "clf"


def load_california():
    p = os.path.join(DATA, "california_housing.csv")
    if not os.path.exists(p):
        from sklearn.datasets import fetch_california_housing
        fetch_california_housing(as_frame=True).frame.to_csv(p, index=False)
    df = pd.read_csv(p)
    return df.drop(columns="MedHouseVal"), df["MedHouseVal"], "reg"


def load_bike():
    df = pd.read_csv(os.path.join(DATA, "day.csv"))
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])
    df = pd.get_dummies(df, columns=["season", "mnth", "weekday", "weathersit"], drop_first=True)
    df = _bool_to_int(df)
    return df.drop(columns="cnt"), df["cnt"], "reg"


def load_ames():
    df = pd.read_csv(os.path.join(DATA, "AmesHousing.csv"))
    df = df.drop(columns=[c for c in ("Order", "PID", "Id") if c in df.columns])
    df = df.dropna(axis=1, thresh=int(len(df) * 0.8)).dropna()
    y = df.pop("SalePrice")
    X = _bool_to_int(pd.get_dummies(df, drop_first=True))
    return X, y, "reg"


DATASETS = {"wine": load_wine, "insurance": load_insurance,
            "adult": load_adult, "bank": load_bank, "diabetes": load_diabetes,
            "california": load_california, "bike": load_bike, "ames": load_ames}


def split(name, seed=SEED):
    X, y, task = DATASETS[name]()
    X = X.astype(float)
    strat = y if task == "clf" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    if len(Xtr) > MAX_TRAIN:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, random_state=seed,
                                          stratify=ytr if strat is not None else None)
    return (Xtr.reset_index(drop=True), Xte.reset_index(drop=True),
            ytr.reset_index(drop=True), yte.reset_index(drop=True), task)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class TabNetWrap(BaseEstimator):
    """sklearn-style wrapper so TabNet works inside a Pipeline and with DataFrames."""
    def __init__(self, task="clf", max_epochs=300, patience=30, seed=SEED):
        self.task, self.max_epochs, self.patience, self.seed = task, max_epochs, patience, seed

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        # small datasets: smaller batches so each epoch has several optimisation steps
        bs = int(min(1024, max(64, len(X) // 8)))
        vbs = int(min(128, bs // 2))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X)); nv = max(1, len(X) // 10)
        va, tr = idx[:nv], idx[nv:]
        kw = dict(max_epochs=self.max_epochs, patience=self.patience, batch_size=bs, virtual_batch_size=vbs)
        if self.task == "clf":
            self.m_ = TabNetClassifier(verbose=0, seed=self.seed)
            self.m_.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **kw)
            self.classes_ = self.m_.classes_
        else:
            y = y.astype(np.float32).reshape(-1, 1)
            self.m_ = TabNetRegressor(verbose=0, seed=self.seed)
            self.m_.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **kw)
        return self

    def predict(self, X):
        p = self.m_.predict(np.asarray(X, dtype=np.float32))
        return p.ravel() if self.task == "reg" else p

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, dtype=np.float32))


THREADS = int(os.environ.get("MODEL_THREADS", 2))
SVM_MAX_TRAIN = 5000       # RBF-SVM prediction cost grows with #support vectors; documented in the paper


def make_model(name, task, seed=SEED):
    from xgboost import XGBClassifier, XGBRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor
    clf = task == "clf"
    est = {
        "RF":  (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=300, n_jobs=THREADS, random_state=seed),
        "XGB": (XGBClassifier if clf else XGBRegressor)(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                         n_jobs=THREADS, random_state=seed),
        "CAT": (CatBoostClassifier if clf else CatBoostRegressor)(iterations=500, verbose=0, random_seed=seed,
                                                                   thread_count=THREADS),
        "ANN": (MLPClassifier if clf else MLPRegressor)(hidden_layer_sizes=(64, 32), max_iter=500,
                                                         early_stopping=True, random_state=seed),
        "SVM": SVC(probability=True, random_state=seed) if clf else SVR(kernel="rbf"),
        "TabNet": TabNetWrap(task, seed=seed),
    }[name]
    return Pipeline([("scale", StandardScaler()), ("est", est)])


def fit_model(name, task, Xtr, ytr, seed=SEED):
    if name == "SVM" and len(Xtr) > SVM_MAX_TRAIN:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=SVM_MAX_TRAIN, random_state=seed,
                                          stratify=ytr if task == "clf" else None)
    m = make_model(name, task, seed)
    yfit = ytr
    if task == "clf" and name == "XGB":            # xgboost needs 0..K-1 labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(ytr)
        m.fit(Xtr, le.transform(ytr))
        m.label_encoder_ = le
    elif task == "reg" and name in ("ANN", "SVM", "TabNet"):   # scale target for NN/SVM regressors
        from sklearn.compose import TransformedTargetRegressor
        m = TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, yfit)
    return m


def score(m, task, Xte, yte):
    from sklearn.metrics import accuracy_score, r2_score
    if task == "clf":
        pred = m.predict(Xte)
        if hasattr(m, "label_encoder_"):
            pred = m.label_encoder_.inverse_transform(pred)
        return accuracy_score(yte, pred)
    return r2_score(yte, m.predict(Xte))


MODELS = ["RF", "XGB", "CAT", "ANN", "SVM", "TabNet"]


def background(Xtr, n=BACKGROUND_N, seed=SEED):
    return Xtr.sample(min(n, len(Xtr)), random_state=seed).reset_index(drop=True)


def save(df, name):
    p = os.path.join(RESULTS, name)
    df.to_csv(p, index=False)
    print(f"[saved] {p}  ({len(df)} rows)", flush=True)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
