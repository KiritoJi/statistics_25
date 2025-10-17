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
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

from common.preprocess import Preprocessor, WOEEncoder, KFoldTargetEncoder
from common.evaluation import compute_metrics, find_best_threshold_fbeta, plot_calibration, plot_feature_importance, save_json

class LogisticWOEModel:
    def __init__(self,
                 encoder_type: str = 'woe',  # 'woe' or 'target'
                 calibrate_method: str = 'sigmoid',  # 'sigmoid' or 'isotonic'
                 use_smote: bool = True,
                 random_state: int = 42):
        self.encoder_type = encoder_type
        self.calibrate_method = calibrate_method
        self.use_smote = use_smote
        self.random_state = random_state
        self.feature_names_: list = []

    def _encode(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cat_cols = [c for c in ['housing','purpose'] if c in X_train.columns]
        if self.encoder_type == 'target':
            enc = KFoldTargetEncoder(cols=cat_cols, n_splits=5, alpha=10.0)
            enc.fit(X_train, y_train)
            Xt = enc.transform(X_train)
            Xv = enc.transform(X_valid)
        else:
            enc = WOEEncoder(cols=cat_cols, alpha=0.5)
            enc.fit(X_train, y_train)
            Xt = enc.transform(X_train)
            Xv = enc.transform(X_valid)
        return Xt, Xv

    def run_cv(self, train_df: pd.DataFrame, out_dir: str) -> Dict:
        df = train_df.copy()
        y = df['target'].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)
        pre = Preprocessor(cat_cols=cat_cols, num_cols=[c for c in num_cols if c != 'target'])
        pre.fit(df)
        dfp = pre.transform(df)
        # 准备特征列表：去除 target
        feat_cols = [c for c in dfp.columns if c != 'target']
        self.feature_names_ = feat_cols
        X = dfp[feat_cols]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_raw = np.zeros(len(df))
        oof_cal = np.zeros(len(df))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y[tr_idx], y[va_idx]
            # 编码（折内拟合）
            X_tr_enc, X_va_enc = self._encode(X_tr, pd.Series(y_tr), X_va)
            # 标准化
            scaler = StandardScaler()
            X_tr_enc[self.feature_names_] = scaler.fit_transform(X_tr_enc[self.feature_names_])
            X_va_enc[self.feature_names_] = scaler.transform(X_va_enc[self.feature_names_])
            # SMOTE（仅训练折）
            if self.use_smote:
                sm = SMOTE(random_state=self.random_state)
                X_tr_enc, y_tr = sm.fit_resample(X_tr_enc, y_tr)
            # 基学习器
            base = LogisticRegression(
                class_weight='balanced',
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=500,
                random_state=self.random_state
            )
            base.fit(X_tr_enc, y_tr)
            # 原始概率
            oof_raw[va_idx] = base.predict_proba(X_va_enc)[:, 1]
            # 校准（折内：对训练折进行内部CV校准，再用于验证折）
            cal = CalibratedClassifierCV(estimator=base, method=self.calibrate_method, cv=3)
            cal.fit(X_tr_enc, y_tr)
            oof_cal[va_idx] = cal.predict_proba(X_va_enc)[:, 1]
        # 评估
        metrics_raw = compute_metrics(y, oof_raw)
        metrics_cal = compute_metrics(y, oof_cal)
        best = find_best_threshold_fbeta(y, oof_cal, beta=2.0)
        # 校准曲线
        plot_calibration(y, oof_raw, oof_cal,
                         out_path=f"{out_dir}/fig_calibration_logistic.png",
                         title='逻辑回归 概率校准曲线（前后对比）')
        # 系数重要性（用全量拟合一次）
        # 全量编码与标准化
        X_full = X.copy()
        enc_full = WOEEncoder(cols=cat_cols, alpha=0.5) if self.encoder_type=='woe' else KFoldTargetEncoder(cols=cat_cols, n_splits=5)
        enc_full.fit(X_full, pd.Series(y))
        X_full = enc_full.transform(X_full)
        scaler_full = StandardScaler()
        X_full[self.feature_names_] = scaler_full.fit_transform(X_full[self.feature_names_])
        lr_full = LogisticRegression(class_weight='balanced', C=1.0, penalty='l2', solver='liblinear', max_iter=500, random_state=self.random_state)
        lr_full.fit(X_full, y)
        coef_imp = pd.DataFrame({'feature': self.feature_names_, 'importance': np.abs(lr_full.coef_).ravel()})
        plot_feature_importance(coef_imp, f"{out_dir}/fig_feature_importance_logistic.png", title='逻辑回归 系数重要性')
        # 保存指标
        result = {
            'model': 'logistic',
            'metrics_raw': metrics_raw,
            'metrics_calibrated': metrics_cal,
            'best_threshold_f2': best
        }
        save_json(result, f"{out_dir}/metrics_logistic.json")
        return {'oof_raw': oof_raw, 'oof_cal': oof_cal, 'y': y, 'result': result}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> str:
        # 统一预处理（使用训练集中位数等）
        df = train_df.copy()
        y = df['target'].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)
        pre = Preprocessor(cat_cols=cat_cols, num_cols=[c for c in num_cols if c != 'target'])
        pre.fit(df)
        dfp = pre.transform(df)
        testp = pre.transform(test_df)
        feat_cols = [c for c in dfp.columns if c != 'target']
        # 编码（用全训练集拟合）
        if self.encoder_type == 'target':
            enc = KFoldTargetEncoder(cols=cat_cols, n_splits=5, alpha=10.0)
        else:
            enc = WOEEncoder(cols=cat_cols, alpha=0.5)
        enc.fit(dfp[feat_cols], pd.Series(y))
        X_full = enc.transform(dfp[feat_cols])
        X_test = enc.transform(testp[feat_cols])
        # 标准化
        scaler = StandardScaler()
        X_full[self.feature_names_] = scaler.fit_transform(X_full[self.feature_names_])
        X_test[self.feature_names_] = scaler.transform(X_test[self.feature_names_])
        # 模型 + 校准（全量）
        base = LogisticRegression(class_weight='balanced', C=1.0, penalty='l2', solver='liblinear', max_iter=500, random_state=self.random_state)
        cal = CalibratedClassifierCV(estimator=base, method=self.calibrate_method, cv=5)
        cal.fit(X_full, y)
        proba = cal.predict_proba(X_test)[:, 1]
        sub = pd.DataFrame({'id': test_df['id'].astype(int), 'target': proba})
        sub_path = f"{out_dir}/submission_logistic.csv"
        sub.to_csv(sub_path, index=False)
        return sub_path
