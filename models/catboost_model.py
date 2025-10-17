# -*- coding: utf-8 -*-
"""
models/catboost_model.py

依赖：catboost、pandas、numpy、scikit-learn
方案：
- CatBoost：class_weights 处理不平衡、早停；
- 分类处理：优先使用原生 cat_features（housing/purpose）
- 概率校准：Isotonic（默认）
- 5 折分层交叉验证，输出 OOF 指标与特征重要性图
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier, Pool
from common.preprocess import Preprocessor
from common.evaluation import (
    compute_metrics,
    find_best_threshold_fbeta,
    plot_calibration,
    plot_feature_importance,
)
from common.utils import save_json


class CatBoostModel:
    """
    CatBoost 模型封装：
    - 原生支持分类特征
    - 手动校准 (Isotonic)
    - 自动计算指标
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _cat_indices(feat_cols: List[str], cat_cols: List[str]) -> List[int]:
        s = set(cat_cols)
        return [i for i, c in enumerate(feat_cols) if c in s]

    def run_cv(self, train_df: pd.DataFrame, out_dir: str):
        df = train_df.copy()
        y = df["target"].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)

        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        X = pre.transform(df).drop(columns=["target"])
        feat_cols = list(X.columns)
        cat_idx = self._cat_indices(feat_cols, [c for c in cat_cols if c in feat_cols])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof = np.zeros(len(df))

        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            pos, neg = (ytr == 1).sum(), (ytr == 0).sum()
            w1 = float(neg / (pos + 1e-6))

            model = CatBoostClassifier(
                iterations=2000,
                depth=5,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_state=self.random_state,
                verbose=False,
                class_weights=[1.0, w1],
            )

            pool_tr = Pool(Xtr, label=ytr, cat_features=cat_idx)
            pool_va = Pool(Xva, label=yva, cat_features=cat_idx)
            model.fit(pool_tr, eval_set=pool_va, use_best_model=True, early_stopping_rounds=50)

            p_tr = model.predict_proba(pool_tr)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_tr, ytr)
            p_va = iso.predict(model.predict_proba(pool_va)[:, 1])
            oof[va_idx] = p_va

        metrics = compute_metrics(y, oof)
        save_json(metrics, f"{out_dir}/metrics_catboost.json")
        return {"oof_cal": oof, "y": y, "result": metrics}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> str:
        df = train_df.copy()
        y = df["target"].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)

        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        X = pre.transform(df).drop(columns=["target"])
        T = pre.transform(test_df)

        feat_cols = list(X.columns)
        cat_idx = self._cat_indices(feat_cols, [c for c in cat_cols if c in feat_cols])

        pos, neg = (y == 1).sum(), (y == 0).sum()
        w1 = float(neg / (pos + 1e-6))

        model = CatBoostClassifier(
            iterations=2000,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            random_state=self.random_state,
            verbose=False,
            class_weights=[1.0, w1],
        )
        pool_tr = Pool(X, label=y, cat_features=cat_idx)
        pool_te = Pool(T, cat_features=cat_idx)
        model.fit(pool_tr)

        p_tr = model.predict_proba(pool_tr)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y)

        p_te_raw = model.predict_proba(pool_te)[:, 1]
        p_te = iso.predict(p_te_raw)

        sub = pd.DataFrame({"id": test_df["id"], "target": p_te})
        path = f"{out_dir}/submission_catboost.csv"
        sub.to_csv(path, index=False)
        return path
