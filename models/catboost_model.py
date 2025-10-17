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
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool

from common.preprocess import Preprocessor
from common.evaluation import compute_metrics, find_best_threshold_fbeta, plot_calibration, plot_feature_importance, save_json

class CatBoostModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names_: list = []
        self.cat_features_idx_: list = []

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
        # cat_features 索引
        self.cat_features_idx_ = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_raw = np.zeros(len(df))
        oof_cal = np.zeros(len(df))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y[tr_idx], y[va_idx]
            pos = (y_tr==1).sum(); neg = (y_tr==0).sum()
            w0 = float(1.0); w1 = float(neg/(pos+1e-6))
            model = CatBoostClassifier(
                iterations=3000,
                learning_rate=0.03,
                depth=5,
                l2_leaf_reg=3.0,
                loss_function='Logloss',
                random_state=self.random_state,
                eval_metric='AUC',
                verbose=False,
                class_weights=[w0, w1]
            )
            train_pool = Pool(X_tr, label=y_tr, cat_features=self.cat_features_idx_)
            valid_pool = Pool(X_va, label=y_va, cat_features=self.cat_features_idx_)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=50)
            # 原始概率
            proba_va = model.predict_proba(valid_pool)[:, 1]
            oof_raw[va_idx] = proba_va
            # 手工 isotonic 校准：基于训练折的 in-sample 预测概率（有限样本下的实用折衷）
            from sklearn.isotonic import IsotonicRegression
            proba_tr = model.predict_proba(train_pool)[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(proba_tr, y_tr)
            oof_cal[va_idx] = iso.predict(proba_va)
        metrics_raw = compute_metrics(y, oof_raw)
        metrics_cal = compute_metrics(y, oof_cal)
        best = find_best_threshold_fbeta(y, oof_cal, beta=2.0)
        plot_calibration(y, oof_raw, oof_cal,
                         out_path=f"{out_dir}/fig_calibration_catboost.png",
                         title='CatBoost 概率校准曲线（前后对比）')
        # 特征重要性（全量拟合）
        pos = (y==1).sum(); neg = (y==0).sum(); w1 = float(neg/(pos+1e-6))
        model_full = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
                                        loss_function='Logloss', random_state=self.random_state, verbose=False, class_weights=[1.0, w1])
        pool_full = Pool(X, label=y, cat_features=self.cat_features_idx_)
        model_full.fit(pool_full)
        imp_vals = model_full.get_feature_importance(pool_full)
        imp = pd.DataFrame({'feature': self.feature_names_, 'importance': imp_vals})
        plot_feature_importance(imp, f"{out_dir}/fig_feature_importance_catboost.png", title='CatBoost 特征重要性')
        result = {
            'model': 'catboost',
            'metrics_raw': metrics_raw,
            'metrics_calibrated': metrics_cal,
            'best_threshold_f2': best
        }
        save_json(result, f"{out_dir}/metrics_catboost.json")
        return {'oof_raw': oof_raw, 'oof_cal': oof_cal, 'y': y, 'result': result}

    def fit_predict_test(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> str:
        df = train_df.copy(); y = df['target'].astype(int).values
        cat_cols, num_cols = Preprocessor.suggest_columns(df)
        pre = Preprocessor(cat_cols=cat_cols, num_cols=[c for c in num_cols if c != 'target'])
        pre.fit(df)
        dfp = pre.transform(df)
        testp = pre.transform(test_df)
        feat_cols = [c for c in dfp.columns if c != 'target']
        self.feature_names_ = feat_cols
        X_full = dfp[feat_cols]
        X_test = testp[feat_cols]
        self.cat_features_idx_ = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
        pos = (y==1).sum(); neg = (y==0).sum(); w1 = float(neg/(pos+1e-6))
        model = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
                                   loss_function='Logloss', random_state=self.random_state, verbose=False, class_weights=[1.0, w1])
        # 使用全量训练拟合（原生分类）
        pool_full = Pool(X_full, label=y, cat_features=self.cat_features_idx_)
        model.fit(pool_full)
        test_pool = Pool(X_test, cat_features=self.cat_features_idx_)
        proba = model.predict_proba(test_pool)[:, 1]
        sub = pd.DataFrame({'id': test_df['id'].astype(int), 'target': proba})
        sub_path = f"{out_dir}/submission_catboost.csv"
        sub.to_csv(sub_path, index=False)
        return sub_path
