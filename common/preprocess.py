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
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# ============ 编码器实现 ============
class WOEEncoder:
    """按类别计算 WOE: ln( (bad_i / good_i) / (bad / good) )，使用平滑避免 0
    仅用于二分类场景。
    """
    def __init__(self, cols: List[str], alpha: float = 0.5):
        self.cols = cols
        self.alpha = alpha
        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.global_ratio_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df['__y__'] = y.values
        bad = df['__y__'].sum()
        good = len(df) - bad
        global_ratio = (bad + self.alpha) / (good + self.alpha)
        for c in self.cols:
            grp = df.groupby(c)['__y__'].agg(['sum', 'count'])
            bad_i = grp['sum']
            good_i = grp['count'] - grp['sum']
            woe = np.log(((bad_i + self.alpha) / (good_i + self.alpha)) / global_ratio)
            self.mapping_[c] = woe.to_dict()
            self.global_ratio_[c] = global_ratio
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        for c in self.cols:
            m = self.mapping_.get(c, {})
            default_woe = 0.0
            X_enc[c] = X_enc[c].map(lambda v: m.get(v, default_woe))
        return X_enc

class KFoldTargetEncoder:
    """K折目标编码，折内拟合避免泄露。建议仅在外部CV中使用：
    - 对训练折：根据训练折拟合，再转换训练折与验证折
    - 对测试集：用全训练集拟合后转换
    支持平滑：enc = (sum_y + alpha * global_mean) / (cnt + alpha)
    """
    def __init__(self, cols: List[str], n_splits: int = 5, alpha: float = 10.0):
        self.cols = cols
        self.n_splits = n_splits
        self.alpha = alpha
        self.global_mean_: Dict[str, float] = {}
        self.mapping_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df['__y__'] = y.values
        for c in self.cols:
            g = df.groupby(c)['__y__'].agg(['sum', 'count'])
            global_mean = df['__y__'].mean()
            enc = (g['sum'] + self.alpha * global_mean) / (g['count'] + self.alpha)
            self.mapping_[c] = enc.to_dict()
            self.global_mean_[c] = global_mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        for c in self.cols:
            m = self.mapping_.get(c, {})
            default_val = self.global_mean_.get(c, 0.0)
            X_enc[c] = X_enc[c].map(lambda v: m.get(v, default_val))
        return X_enc

class LabelEncoderWrapper:
    """简单标签编码，未见类别映射为 -1。用于树模型。"""
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.mapping_: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame):
        for c in self.cols:
            cats = pd.Series(X[c].astype(str).unique())
            self.mapping_[c] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        for c in self.cols:
            m = self.mapping_.get(c, {})
            Xt[c] = Xt[c].astype(str).map(lambda v: m.get(v, -1)).astype(int)
        return Xt

# ============ 预处理封装 ============
class Preprocessor:
    def __init__(self,
                 cat_cols: Optional[List[str]] = None,
                 num_cols: Optional[List[str]] = None,
                 log1p_cols: Optional[List[str]] = None):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.log1p_cols = log1p_cols or []
        self.medians_: Dict[str, float] = {}

    @staticmethod
    def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        # 比率与组合特征（存在即计算）
        if {'credict_used_amount', 'credict_limit'}.issubset(X.columns):
            util = np.where(X['credict_limit'] > 0,
                            X['credict_used_amount'] / X['credict_limit'], 0.0)
            X['credit_utilization'] = np.clip(util, 0, 5)
        if {'total_balance', 'income'}.issubset(X.columns):
            X['balance_to_income'] = np.where(X['income'] > 0,
                                              X['total_balance'] / X['income'], 0.0)
        if {'amount', 'income'}.issubset(X.columns):
            X['amount_to_income'] = np.where(X['income'] > 0,
                                             X['amount'] / X['income'], 0.0)
        if {'inquire_times', 'overdue_times'}.issubset(X.columns):
            X['query_to_overdue'] = X['inquire_times'] / (X['overdue_times'] + 1.0)
        # 近期活跃刻画
        if {'recent_loan_number', 'recent_account_months'}.issubset(X.columns):
            X['recent_loan_rate'] = X['recent_loan_number'] / (X['recent_account_months'] + 1.0)
        if 'last_credict_card_months' in X.columns:
            X['last_card_recency'] = 1.0 / (X['last_credict_card_months'] + 1.0)
        # 信用时长：直接使用 length（若为贷款期或工龄，这里照用）
        # 偏态 log1p
        for c in ['amount','income','total_balance','credict_used_amount','credict_limit',
                  'inquire_times','default_times','total_default_number','last_overdue_months']:
            if c in X.columns:
                X[f'log1p_{c}'] = np.log1p(np.clip(X[c].astype(float), a_min=0, a_max=None))
        return X

    def fit(self, df: pd.DataFrame):
        X = df.copy()
        # 缺失：分类填"未知"
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(str).fillna('未知')
        # 缺失：数值中位数
        for c in self.num_cols:
            if c in X.columns:
                self.medians_[c] = pd.to_numeric(X[c], errors='coerce').median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(str).fillna('未知')
        for c in self.num_cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(self.medians_.get(c, 0.0))
        X = self._derive_features(X)
        # 最终选择：数值 + 编码后分类（编码在模型侧完成，这里只保留原列）
        return X

    @staticmethod
    def suggest_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """根据已知数据推断数值/分类列"""
        cat_cols = [c for c in ['housing','purpose'] if c in df.columns]
        num_cols = [c for c in df.columns if c not in cat_cols + ['target']]
        return cat_cols, num_cols

# ============ 数据加载 ============
def load_train_test(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_excel(train_path, engine='openpyxl')
    test_df = pd.read_excel(test_path, engine='openpyxl')
    return train_df, test_df
