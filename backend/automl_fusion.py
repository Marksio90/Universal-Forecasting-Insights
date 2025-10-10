from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np, pandas as pd, optuna, warnings, os, mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, VotingRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

@dataclass
class FusionResult:
    problem_type: str
    metric_name: str
    best_score: float
    leaderboard: List[Tuple[str,float]]
    blend_weights: Dict[str,float] | None
    model: Any

def _infer_type(y: pd.Series)->str:
    if pd.api.types.is_float_dtype(y): return "regression"
    return "classification" if y.nunique()<=50 else "regression"

def _build_preprocessor(X: pd.DataFrame)->ColumnTransformer:
    num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat=[c for c in X.columns if c not in num]
    return ColumnTransformer([("num", StandardScaler(with_mean=False), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

def _split(X,y,seed):
    strat = y if y.nunique()<=20 and not pd.api.types.is_float_dtype(y) else None
    return train_test_split(X,y,test_size=0.2,random_state=seed, stratify=strat)

def _metric_cls(y_true, y_pred): 
    return "f1", float(f1_score(y_true, y_pred, average="weighted"))
def _metric_reg(y_true, y_pred): 
    return "r2", float(r2_score(y_true, y_pred))

def _smote_maybe(is_cls, pre, model):
    if is_cls:
        try: return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("model", model)])
        except Exception: ...
    return Pipeline([("pre", pre), ("model", model)])

def train_fusion(df: pd.DataFrame, target: str, *, seed:int=42, trials:int=30)->FusionResult:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
    with mlflow.start_run() as run:
        y=df[target]; X=df.drop(columns=[target])
        problem=_infer_type(y); is_cls = problem=="classification"
        pre=_build_preprocessor(X)
        X_tr, X_te, y_tr, y_te = _split(X,y,seed)

        def objective(trial: optuna.Trial):
            models = []
            if is_cls:
                models += [
                    ("XGB", XGBClassifier(n_estimators=trial.suggest_int("xgb_n",300,900), tree_method="hist", eval_metric="logloss")),
                    ("LGBM", LGBMClassifier(n_estimators=trial.suggest_int("lgbm_n",300,900))),
                    ("CAT", CatBoostClassifier(iterations=trial.suggest_int("cat_n",300,900), verbose=False))
                ]
            else:
                models += [
                    ("XGB", XGBRegressor(n_estimators=trial.suggest_int("xgb_n",300,900), tree_method="hist")),
                    ("LGBM", LGBMRegressor(n_estimators=trial.suggest_int("lgbm_n",300,900))),
                    ("CAT", CatBoostRegressor(iterations=trial.suggest_int("cat_n",300,900), verbose=False))
                ]
            best=-1e9
            for name, model in models:
                pipe=_smote_maybe(is_cls, pre, model); pipe.fit(X_tr, y_tr)
                pred=pipe.predict(X_te)
                _, score = (_metric_cls(y_te, pred) if is_cls else _metric_reg(y_te, pred))
                trial.set_user_attr(name, float(score)); best=max(best, float(score))
            return best

        study=optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        scores_map={}
        for t in study.trials:
            for k,v in t.user_attrs.items(): scores_map.setdefault(k, []).append(v)
        leader = sorted([(k, float(sum(v)/len(v))) for k,v in scores_map.items()], key=lambda x: x[1], reverse=True)[:3]
        names, means = zip(*leader)
        weights = [float(m / (sum(means)+1e-9)) for m in means]
        est=[]
        for name in names:
            if problem=="classification":
                m = XGBClassifier(n_estimators=600, tree_method="hist") if name=="XGB" else (LGBMClassifier(n_estimators=700) if name=="LGBM" else CatBoostClassifier(iterations=600, verbose=False))
            else:
                m = XGBRegressor(n_estimators=800, tree_method="hist") if name=="XGB" else (LGBMRegressor(n_estimators=800) if name=="LGBM" else CatBoostRegressor(iterations=700, verbose=False))
            est.append((name, _smote_maybe(problem=="classification", _build_preprocessor(X), m)))
        final = (VotingClassifier(estimators=est, voting="soft", weights=weights) if problem=="classification"
                 else VotingRegressor(estimators=est, weights=weights))
        final.fit(X, y)
        mlflow.log_params({"problem": problem, "top_models": ",".join(names)})
        mlflow.log_metrics({("f1" if problem=="classification" else "r2"): float(max(means))})
        return FusionResult(problem, "f1" if problem=="classification" else "r2", float(max(means)), leader, {n:w for n,w in zip(names, weights)}, final)
