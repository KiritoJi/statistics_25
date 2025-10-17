# -*- coding: utf-8 -*-
"""
models/logistic_woe.py

依赖：pandas、numpy、scikit-learn、imbalanced-learn
方案：
- 逻辑回归（class_weight='balanced'，L2 正则）
- 分类编码：housing/purpose 使用 WOE 或 K折目标编码（参数可选，默认 WOE）
- SMOTE：仅对训练折启用（可选）
- 概率校准：Platt（sigmoid）或 Isotonic（默认 sigmoid）
- 5 折分层交叉验证，输出 OOF 指标与校准曲线、特征重要性（系数）
"""
# -*- coding: utf-8 -*-
"""
models/logistic_woe.py
逻辑回归 + WOE 编码 + 可选 SMOTE + 概率校准
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from common.preprocess import Preprocessor
from common.evaluation import compute_metrics
from common.utils import save_json


class WOEEncoder:
    """简单 WOE 编码实现（按二分类标签统计）"""
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.woe_maps = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        for col in self.cols:
            df = pd.DataFrame({"x": X[col], "y": y})
            stats = df.groupby("x")["y"].agg(["mean", "count"])
            eps = 1e-6
            stats["woe"] = np.log(
                ((1 - stats["mean"]) + eps) / (stats["mean"] + eps)
            )
            self.woe_maps[col] = stats["woe"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_woe = X.copy()
        for col in self.cols:
            if col in self.woe_maps:
                X_woe[col] = X_woe[col].map(self.woe_maps[col]).fillna(0)
        return X_woe


class LogisticWOEModel:
    def __init__(
        self,
        encoder_type: str = "woe",
        calibrate_method: str = "sigmoid",
        use_smote: bool = False,
        random_state: int = 42,
    ):
        """
        参数说明：
        encoder_type: 'woe' 或 'none'
        calibrate_method: 'sigmoid' 或 'isotonic'
        use_smote: 是否启用SMOTE过采样
        """
        self.encoder_type = encoder_type
        self.calibrate_method = calibrate_method
        self.use_smote = use_smote
        self.random_state = random_state

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
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            # ============ 编码 ============
            if self.encoder_type == "woe" and len(cat_cols) > 0:
                enc = WOEEncoder(cat_cols).fit(Xtr, ytr)
                Xtr = enc.transform(Xtr)
                Xva = enc.transform(Xva)

            # ============ SMOTE ============
            if self.use_smote:
                sm = SMOTE(random_state=self.random_state)
                Xtr, ytr = sm.fit_resample(Xtr, ytr)

            # ============ 训练 ============
            scaler = StandardScaler()
            Xtr_scaled = scaler.fit_transform(Xtr)
            Xva_scaled = scaler.transform(Xva)

            base = LogisticRegression(
                solver="lbfgs", max_iter=2000, class_weight="balanced"
            )
            clf = CalibratedClassifierCV(
                base, method=self.calibrate_method, cv=3
            )
            clf.fit(Xtr_scaled, ytr)
            oof[va_idx] = clf.predict_proba(Xva_scaled)[:, 1]

        metrics = compute_metrics(y, oof)
        save_json(metrics, f"{out_dir}/metrics_logistic_woe.json")
        return {"oof_cal": oof, "y": y, "result": metrics}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
        df = train_df.copy()
        y = df["target"].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)

        pre = Preprocessor(cat_cols, num_cols)
        pre.fit(df)
        X = pre.transform(df).drop(columns=["target"])
        T = pre.transform(test_df)

        if self.encoder_type == "woe" and len(cat_cols) > 0:
            enc = WOEEncoder(cat_cols).fit(X, y)
            X = enc.transform(X)
            T = enc.transform(T)

        if self.use_smote:
            sm = SMOTE(random_state=self.random_state)
            X, y = sm.fit_resample(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        T_scaled = scaler.transform(T)

        base = LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced"
        )
        clf = CalibratedClassifierCV(base, method=self.calibrate_method, cv=5)
        clf.fit(X_scaled, y)
        p = clf.predict_proba(T_scaled)[:, 1]

        sub = pd.DataFrame({"id": test_df["id"], "target": p})
        path = f"{out_dir}/submission_logistic_woe.csv"
        sub.to_csv(path, index=False)
        return path
