# -*- coding: utf-8 -*-
"""
common/evaluation.py

依赖：numpy、pandas、matplotlib、seaborn、scikit-learn
功能：
- 统一评估指标 AUC、PR-AUC、Brier
- F-beta(beta=2) 阈值搜索
- 关键可视化：标签占比、模型对比、校准曲线、特征重要性条形图
- LightGBM 的 SHAP 重要性绘制（单独函数）

注意：
- 中文字体：Noto Sans CJK SC
- 图例统一放在底部，预留足够边距，避免与坐标轴/标题重叠
"""
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_fscore_support
from sklearn.calibration import calibration_curve

# 字体设置（确保中文显示正常）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']

# ============ 指标 ============
def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float('nan')
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    return {'auc': float(auc), 'pr_auc': float(pr_auc), 'brier': float(brier)}


def find_best_threshold_fbeta(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> Dict[str, float]:
    thresholds = np.unique(np.quantile(y_proba, np.linspace(0, 1, 200)))
    best = {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'fbeta': 0.0}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, average='binary', zero_division=0)
        if f > best['fbeta']:
            best = {'threshold': float(t), 'precision': float(p), 'recall': float(r), 'fbeta': float(f)}
    return best

# ============ 可视化 ============
def plot_label_balance(y: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    pos = np.sum(y)
    neg = len(y) - pos
    ax.bar(['正常(0)','违约(1)'], [neg, pos], color=['#4C78A8','#F58518'])
    ax.set_title('标签占比')
    ax.set_ylabel('样本数')
    plt.xticks(rotation=0)
    plt.legend(['负样本','正样本'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.3, top=0.88)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    metrics = ['pr_auc','auc','brier']
    # 归一化 brier 为可视化（越小越好，图表显示取(1-brier_norm)）
    df = summary_df.copy()
    max_brier = df['brier'].max()
    df['brier_display'] = 1 - df['brier'] / (max_brier + 1e-9)
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['pr_auc'], width, label='PR-AUC')
    ax.bar(x, df['auc'], width, label='AUC')
    ax.bar(x + width, df['brier_display'], width, label='(1 - Brier归一化)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.set_ylabel('指标值')
    ax.set_title('模型对比（PR-AUC/AUC/Brier）')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.35, top=0.88)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(feat_imp: pd.DataFrame, out_path: str, top_k: int = 20, title: str = '特征重要性'):
    df = feat_imp.sort_values('importance', ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.barh(df['feature'][::-1], df['importance'][::-1], color='#4C78A8')
    ax.set_xlabel('重要性')
    ax.set_title(title)
    plt.legend(['特征重要性'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    fig.subplots_adjust(bottom=0.3, left=0.28, top=0.88)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_calibration(y_true: np.ndarray,
                     y_raw: np.ndarray,
                     y_cal: np.ndarray,
                     out_path: str,
                     title: str):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_raw, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_cal, n_bins=10)
    ax.plot([0,1],[0,1], linestyle='--', color='grey', label='理想校准')
    ax.plot(prob_pred_raw, prob_true_raw, marker='o', label='校准前')
    ax.plot(prob_pred_cal, prob_true_cal, marker='o', label='校准后')
    ax.set_xlabel('预测概率分箱均值')
    ax.set_ylabel('实际违约率')
    ax.set_title(title)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.35, top=0.88)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def save_json(d: Dict, out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
