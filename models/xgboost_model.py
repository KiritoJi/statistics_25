# -*- coding: utf-8 -*-
"""
models/xgboost_model.py

依赖：xgboost、pandas、numpy、scikit-learn
方案：
- XGBoost：不平衡处理（scale_pos_weight）、早停、正则与采样
- 分类编码：树模型使用标签编码（折内拟合）
- 概率校准：Isotonic（默认）
- 5 折分层交叉验证，输出 OOF 指标与特征重要性图
"""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from common.preprocess import Preprocessor, LabelEncoderWrapper
from common.evaluation import compute_metrics, find_best_threshold_fbeta, plot_calibration, plot_feature_importance, save_json

class XGBoostModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names_: list = []

    def run_cv(self, train_df: pd.DataFrame, out_dir: str) -> Dict:
        df = train_df.copy()
        y = df['target'].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)
        pre = Preprocessor(cat_cols=cat_cols, num_cols=[c for c in num_cols if c != 'target'])
        pre.fit(df)
        dfp = pre.transform(df)
        feat_cols = [c for c in dfp.columns if c != 'target']
        self.feature_names_ = feat_cols
        X = dfp[feat_cols]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_raw = np.zeros(len(df))
        oof_cal = np.zeros(len(df))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y[tr_idx], y[va_idx]
            le = LabelEncoderWrapper(cols=cat_cols)
            le.fit(X_tr)
            X_tr_enc = le.transform(X_tr)
            X_va_enc = le.transform(X_va)
            pos = (y_tr==1).sum(); neg = (y_tr==0).sum(); spw = float(neg)/(pos+1e-6)
            model = xgb.XGBClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective='binary:logistic',
                tree_method='hist',
                random_state=self.random_state,
                n_jobs=-1,
                scale_pos_weight=spw,
                eval_metric='auc'
            )
            model.fit(X_tr_enc, y_tr,
                      eval_set=[(X_va_enc, y_va)])
            oof_raw[va_idx] = model.predict_proba(X_va_enc)[:, 1]
            cal = CalibratedClassifierCV(estimator=model, method='isotonic', cv=3)
            cal.fit(X_tr_enc, y_tr)
            oof_cal[va_idx] = cal.predict_proba(X_va_enc)[:, 1]
        metrics_raw = compute_metrics(y, oof_raw)
        metrics_cal = compute_metrics(y, oof_cal)
        best = find_best_threshold_fbeta(y, oof_cal, beta=2.0)
        plot_calibration(y, oof_raw, oof_cal,
                         out_path=f"{out_dir}/fig_calibration_xgboost.png",
                         title='XGBoost 概率校准曲线（前后对比）')
        # 特征重要性（全量拟合）
        le_full = LabelEncoderWrapper(cols=cat_cols)
        le_full.fit(X)
        X_full = le_full.transform(X)
        pos = (y==1).sum(); neg = (y==0).sum(); spw = float(neg)/(pos+1e-6)
        model_full = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                       objective='binary:logistic', tree_method='hist',
                                       random_state=self.random_state, n_jobs=-1, scale_pos_weight=spw)
        model_full.fit(X_full, y)
        imp = pd.DataFrame({'feature': self.feature_names_, 'importance': model_full.feature_importances_.astype(float)})
        plot_feature_importance(imp, f"{out_dir}/fig_feature_importance_xgboost.png", title='XGBoost 特征重要性')
        result = {
            'model': 'xgboost',
            'metrics_raw': metrics_raw,
            'metrics_calibrated': metrics_cal,
            'best_threshold_f2': best
        }
        save_json(result, f"{out_dir}/metrics_xgboost.json")
        return {'oof_raw': oof_raw, 'oof_cal': oof_cal, 'y': y, 'result': result}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> str:
        df = train_df.copy(); y = df['target'].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)
        pre = Preprocessor(cat_cols=cat_cols, num_cols=[c for c in num_cols if c != 'target'])
        pre.fit(df)
        dfp = pre.transform(df)
        testp = pre.transform(test_df)
        feat_cols = [c for c in dfp.columns if c != 'target']
        le = LabelEncoderWrapper(cols=cat_cols)
        le.fit(dfp[feat_cols])
        X_full = le.transform(dfp[feat_cols])
        X_test = le.transform(testp[feat_cols])
        pos = (y==1).sum(); neg = (y==0).sum(); spw = float(neg)/(pos+1e-6)
        model = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=4,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective='binary:logistic', tree_method='hist', random_state=self.random_state, n_jobs=-1, scale_pos_weight=spw)
        cal = CalibratedClassifierCV(estimator=model, method='isotonic', cv=5)
        cal.fit(X_full, y)
        proba = cal.predict_proba(X_test)[:, 1]
        sub = pd.DataFrame({'id': test_df['id'].astype(int), 'target': proba})
        sub_path = f"{out_dir}/submission_xgboost.csv"
        sub.to_csv(sub_path, index=False)
        return sub_path
