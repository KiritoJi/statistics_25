# -*- coding: utf-8 -*-
"""
common/preprocess.py

依赖：pandas、numpy、scikit-learn
功能：
- 加载训练/测试数据（Excel）
- 缺失值处理（数值：中位数；分类："未知"）
- 特征派生（比率、近期活跃、信用使用率等）
- 偏态变量 log1p 变换（金额与计数类）
- 分类编码器封装：WOEEncoder、KFoldTargetEncoder（避免泄露，需在CV折内拟合）

使用说明：
- 在模型代码中按折调用 Preprocessor.fit_transform(train_df, y) / transform(valid_df) / transform(test_df)
- 对于树模型可使用 LabelEncoderWrapper 简单编码；逻辑回归建议使用 WOE 或 目标编码
"""
# -*- coding: utf-8 -*-
"""
common/preprocess.py
统一数据预处理模块
---------------------------------
功能：
1️⃣ 缺失值填充（数值中位数 / 类别"未知"）
2️⃣ 类别编码（astype('category').cat.codes）
3️⃣ 派生特征（信用利用率、收支比、金额log1p等）
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self,
                 cat_cols: Optional[List[str]] = None,
                 num_cols: Optional[List[str]] = None):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.medians_: Dict[str, float] = {}

    # =========================================================
    # 特征衍生
    # =========================================================
    @staticmethod
    def derive_features(df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        # 信用利用率
        if {'credict_used_amount', 'credict_limit'}.issubset(X.columns):
            X['credit_utilization'] = np.where(
                X['credict_limit'] > 0,
                X['credict_used_amount'] / X['credict_limit'],
                0.0
            )

        # 收支比
        if {'total_balance', 'income'}.issubset(X.columns):
            X['balance_to_income'] = X['total_balance'] / (X['income'] + 1e-6)

        # log1p 变换的金额特征
        for c in ['amount', 'income', 'total_balance', 'credict_used_amount']:
            if c in X.columns:
                X[f'log1p_{c}'] = np.log1p(np.clip(X[c], a_min=0, a_max=None))

        return X

    # =========================================================
    # 拟合（计算中位数）
    # =========================================================
    def fit(self, df: pd.DataFrame):
        for c in self.num_cols:
            self.medians_[c] = pd.to_numeric(df[c], errors='coerce').median()
        return self

    # =========================================================
    # 转换
    # =========================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        # 类别特征编码（字符串 → category → 数字编码）
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("未知")
            X[c] = X[c].astype("category").cat.codes  # 🔥 核心修复：转为 int 编码

        # 数值特征填充
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.medians_.get(c, 0))

        # 派生特征
        X = self.derive_features(X)

        return X

    # =========================================================
    # 自动列建议
    # =========================================================
    @staticmethod
    def suggest_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """自动识别常见类别与数值列"""
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or c in ['housing', 'purpose']]
        num_cols = [c for c in df.columns if c not in cat_cols + ['target']]
        return cat_cols, num_cols
