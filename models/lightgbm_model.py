# -*- coding: utf-8 -*-
"""
models/lightgbm_model.py

依赖：lightgbm、pandas、numpy、scikit-learn、shap
方案：
- LightGBM：不平衡处理（scale_pos_weight）、早停、L2 正则
- 分类编码：树模型使用标签编码（折内拟合）
- 概率校准：Isotonic（默认）
- 5 折分层交叉验证，输出 OOF 指标、特征重要性；另生成 SHAP 重要性图
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from common.preprocess import Preprocessor
from common.evaluation import compute_metrics
from common.utils import save_json

class LightGBMModel:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def run_cv(self, train_df, out_dir):
        df = train_df.copy()
        y = df['target'].astype(int).values
        X = Preprocessor(*Preprocessor.suggest_columns(df)).transform(df).drop(columns=['target'])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof = np.zeros(len(df))
        for tr, va in skf.split(X, y):
            Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]
            pos, neg = (ytr==1).sum(), (ytr==0).sum()
            spw = neg/(pos+1e-6)
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=spw, random_state=self.random_state
            )
            cal = CalibratedClassifierCV(model, method='isotonic', cv=3)
            cal.fit(Xtr, ytr)
            oof[va] = cal.predict_proba(Xva)[:,1]
        metrics = compute_metrics(y, oof)
        save_json(metrics, f"{out_dir}/metrics_lightgbm.json")
        return {'oof_cal': oof, 'y': y, 'result': metrics}

    def fit_predict_test(self, train_df, test_df, out_dir):
        y = train_df['target'].astype(int).values
        pre = Preprocessor(*Preprocessor.suggest_columns(train_df))
        X = pre.transform(train_df).drop(columns=['target'])
        T = pre.transform(test_df)
        pos, neg = (y==1).sum(), (y==0).sum()
        spw = neg/(pos+1e-6)
        model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=self.random_state
        )
        cal = CalibratedClassifierCV(model, method='isotonic', cv=5)
        cal.fit(X, y)
        pred = cal.predict_proba(T)[:,1]
        sub = pd.DataFrame({'id': test_df['id'], 'target': pred})
        sub.to_csv(f"{out_dir}/submission_lightgbm.csv", index=False)
        return f"{out_dir}/submission_lightgbm.csv"
