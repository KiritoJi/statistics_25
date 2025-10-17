# -*- coding: utf-8 -*-
"""
models/extratrees_model.py

依赖：pandas、numpy、scikit-learn
方案：
- ExtraTrees：n_estimators 较大、class_weight='balanced'
- 分类编码：树模型使用标签编码（折内拟合）
- 概率校准：Isotonic（默认）
- 5 折分层交叉验证，输出 OOF 指标与特征重要性图
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from common.preprocess import Preprocessor
from common.evaluation import compute_metrics
from common.utils import save_json


class _SimpleLabelEncoder:
    """Per-column label encoder for tree models. Unseen -> -1."""
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame):
        for c in self.cols:
            cats = pd.Series(X[c].astype(str).fillna("未知").unique())
            self.maps[c] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        for c in self.cols:
            m = self.maps.get(c, {})
            Xt[c] = Xt[c].astype(str).fillna("未知").map(lambda v: m.get(v, -1)).astype(int)
        return Xt


class ExtraTreesModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _encode_cat(self, Xtr: pd.DataFrame, Xva: pd.DataFrame, cat_cols: List[str]):
        enc = _SimpleLabelEncoder(cat_cols).fit(Xtr)
        return enc.transform(Xtr), enc.transform(Xva)

    def run_cv(self, train_df: pd.DataFrame, out_dir: str):
        df = train_df.copy()
        y = df["target"].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)

        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        X = pre.transform(df).drop(columns=["target"])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof = np.zeros(len(df))

        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            ytr, yva = y[tr_idx], y[va_idx]

            Xtr_enc, Xva_enc = self._encode_cat(Xtr, Xva, [c for c in cat_cols if c in X.columns])

            base = ExtraTreesClassifier(
                n_estimators=1200,
                max_features="sqrt",
                bootstrap=False,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
            cal.fit(Xtr_enc, ytr)
            oof[va_idx] = cal.predict_proba(Xva_enc)[:, 1]

        metrics = compute_metrics(y, oof)
        save_json(metrics, f"{out_dir}/metrics_extratrees.json")
        return {"oof_cal": oof, "y": y, "result": metrics}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> str:
        df = train_df.copy()
        y = df["target"].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)

        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        X = pre.transform(df).drop(columns=["target"])
        T = pre.transform(test_df)

        enc = _SimpleLabelEncoder([c for c in cat_cols if c in X.columns]).fit(X)
        X_enc = enc.transform(X)
        T_enc = enc.transform(T)

        base = ExtraTreesClassifier(
            n_estimators=1200,
            max_features="sqrt",
            bootstrap=False,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
        cal.fit(X_enc, y)
        p = cal.predict_proba(T_enc)[:, 1]

        sub = pd.DataFrame({"id": test_df["id"], "target": p})
        path = f"{out_dir}/submission_extratrees.csv"
        sub.to_csv(path, index=False)
        return path
