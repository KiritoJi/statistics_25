# -*- coding: utf-8 -*-
"""
run_all_models.py

依赖：
- pandas、numpy、scikit-learn、imbalanced-learn、lightgbm、xgboost、catboost、shap、matplotlib、seaborn、openpyxl

运行指南：
- 安装依赖（首次运行自动安装）：
    python run_all_models.py
- 该脚本将：
    1) 读取 Excel 数据（训练数据集.xlsx、测试集.xlsx）与提交样例；
    2) 统一预处理与特征工程；
    3) 5 折分层交叉验证训练并评估 5 个模型（Logistic, LightGBM, XGBoost, CatBoost, ExtraTrees）；
    4) 保存模型校准曲线、特征重要性、标签占比与模型对比图；
    5) 生成各模型测试集预测文件 submission_{model}.csv，并按 OOF PR-AUC 选出 submission_best.csv；
    6) 输出每模型 metrics_{model}.json 与综合 metrics_summary.json。
"""
import os
import sys
import subprocess
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 字体（中文）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']

# 依赖确保
REQUIRED = [
    'pandas','numpy','scikit-learn','imbalanced-learn','lightgbm','xgboost','catboost','shap','matplotlib','seaborn','openpyxl'
]

def ensure_deps():
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace('-', '_'))
        except Exception:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], check=True)


def main():
    ensure_deps()
    os.makedirs('output', exist_ok=True)
    # 读取数据
    train_path = '训练数据集.xlsx'
    test_path = '测试集.xlsx'
    sub_sample_path = '提交样例.csv'
    train_df = pd.read_excel(train_path, engine='openpyxl')
    test_df = pd.read_excel(test_path, engine='openpyxl')
    # 标签占比图
    from common.evaluation import plot_label_balance, plot_model_comparison, save_json
    y = train_df['target'].astype(int).values
    plot_label_balance(y, 'output/fig_label_balance.png')

    # 运行各模型
    from models.logistic_woe import LogisticWOEModel
    from models.lightgbm_model import LightGBMModel
    from models.xgboost_model import XGBoostModel
    from models.catboost_model import CatBoostModel
    from models.extratrees_model import ExtraTreesModel

    results: List[Dict] = []
    oof_dict = {}

    # Logistic Regression（WOE + Platt）
    lr = LogisticWOEModel(encoder_type='woe', calibrate_method='sigmoid', use_smote=True)
    r_lr = lr.run_cv(train_df, 'output')
    sub_lr = lr.fit_predict_test(train_df, test_df, 'output')
    results.append({'model':'logistic', **r_lr['result']})
    oof_dict['logistic'] = {'oof': r_lr['oof_cal'], 'y': r_lr['y']}

    # LightGBM（Isotonic）
    lgbm = LightGBMModel()
    r_lgb = lgbm.run_cv(train_df, 'output')
    sub_lgb = lgbm.fit_predict_test(train_df, test_df, 'output')
    results.append({'model':'lightgbm', **r_lgb['result']})
    oof_dict['lightgbm'] = {'oof': r_lgb['oof_cal'], 'y': r_lgb['y']}

    # XGBoost（Isotonic）
    xgbm = XGBoostModel()
    r_xgb = xgbm.run_cv(train_df, 'output')
    sub_xgb = xgbm.fit_predict_test(train_df, test_df, 'output')
    results.append({'model':'xgboost', **r_xgb['result']})
    oof_dict['xgboost'] = {'oof': r_xgb['oof_cal'], 'y': r_xgb['y']}

    # CatBoost（Isotonic）
    cbm = CatBoostModel()
    r_cb = cbm.run_cv(train_df, 'output')
    sub_cb = cbm.fit_predict_test(train_df, test_df, 'output')
    results.append({'model':'catboost', **r_cb['result']})
    oof_dict['catboost'] = {'oof': r_cb['oof_cal'], 'y': r_cb['y']}

    # ExtraTrees（Isotonic）
    etm = ExtraTreesModel()
    r_et = etm.run_cv(train_df, 'output')
    sub_et = etm.fit_predict_test(train_df, test_df, 'output')
    results.append({'model':'extratrees', **r_et['result']})
    oof_dict['extratrees'] = {'oof': r_et['oof_cal'], 'y': r_et['y']}

    # 汇总指标（按校准后）
    rows = []
    for r in results:
        m = r['metrics_calibrated']
        rows.append({'model': r['model'], 'pr_auc': m['pr_auc'], 'auc': m['auc'], 'brier': m['brier'],
                     'best_threshold': r['best_threshold_f2']['threshold'], 'precision': r['best_threshold_f2']['precision'],
                     'recall': r['best_threshold_f2']['recall'], 'f2': r['best_threshold_f2']['fbeta']})
    summary_df = pd.DataFrame(rows)
    summary_df.to_json('output/metrics_summary.json', orient='records')
    plot_model_comparison(summary_df, 'output/fig_model_comparison.png')

    # 选最优模型（PR-AUC 最大）
    best_model = summary_df.sort_values('pr_auc', ascending=False)['model'].iloc[0]
    model_to_file = {
        'logistic': 'output/submission_logistic.csv',
        'lightgbm': 'output/submission_lightgbm.csv',
        'xgboost': 'output/submission_xgboost.csv',
        'catboost': 'output/submission_catboost.csv',
        'extratrees': 'output/submission_extratrees.csv'
    }
    best_path = model_to_file[best_model]
    best_df = pd.read_csv(best_path)
    # 对齐提交样例的列名与顺序
    best_df = best_df[['id','target']]
    best_df.to_csv('output/submission_best.csv', index=False)

    print('完成。最优模型：', best_model)
    print('最优提交文件：output/submission_best.csv')

if __name__ == '__main__':
    main()

