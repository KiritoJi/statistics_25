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
# -*- coding: utf-8 -*-
"""
common/evaluation.py
模型评估与可视化模块（完整版）
---------------------------------
包含：
1️⃣ compute_metrics
2️⃣ plot_label_balance
3️⃣ plot_model_comparison
4️⃣ find_best_threshold_fbeta
5️⃣ plot_calibration
6️⃣ plot_feature_importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
)
from sklearn.calibration import calibration_curve


# =========================================================
# 1️⃣ 模型指标计算
# =========================================================
def compute_metrics(y_true, y_pred):
    """计算常用二分类评估指标"""
    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred > 0.5).astype(int)).ravel()
    ks = abs((tp / (tp + fn + 1e-9)) - (fp / (fp + tn + 1e-9)))
    mcc = ((tp * tn - fp * fn) /
           np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-9))
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ks": ks,
        "mcc": mcc,
    }


# =========================================================
# 2️⃣ 标签分布绘制
# =========================================================
def plot_label_balance(df_or_y, path=None, target_col="target"):
    """绘制标签分布柱状图（兼容 DataFrame / numpy / list）"""
    if isinstance(df_or_y, pd.DataFrame):
        values = df_or_y[target_col].values
    elif isinstance(df_or_y, pd.Series):
        values = df_or_y.values
    elif isinstance(df_or_y, (list, np.ndarray)):
        values = np.array(df_or_y)
    else:
        raise TypeError(f"Unsupported input type: {type(df_or_y)}")

    unique, counts = np.unique(values, return_counts=True)
    props = counts / counts.sum()

    plt.figure(figsize=(5, 4))
    sns.barplot(x=unique, y=props)
    plt.title("Label Balance (Proportion)")
    plt.ylabel("Proportion")
    plt.xlabel("Label")

    for i, v in enumerate(props):
        plt.text(i, v + 0.01, f"{v:.2%}", ha="center")

    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================================================
# 3️⃣ 多模型指标对比绘图
# =========================================================
def plot_model_comparison(df: pd.DataFrame, path: str):
    """绘制不同模型的多指标对比条形图"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.melt(id_vars="model"),
                x="variable", y="value", hue="model")
    plt.title("模型指标对比 (Model Comparison)")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# =========================================================
# 4️⃣ 寻找最优阈值（F-beta）
# =========================================================
def find_best_threshold_fbeta(y_true, y_prob, beta=1.0):
    """在 0~1 阈值范围内搜索 F_beta 得分最高的阈值"""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta)
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f


# =========================================================
# 5️⃣ 模型校准曲线绘制
# =========================================================
def plot_calibration(y_true, y_prob, path=None, n_bins=10):
    """绘制模型预测概率的校准曲线 (Predicted vs True)"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted probability")
    plt.ylabel("True fraction of positives")
    plt.legend()
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================================================
# 6️⃣ 特征重要性绘制
# =========================================================
def plot_feature_importance(feature_importances, feature_names, path=None, top_n=20):
    """绘制特征重要性条形图"""
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(data=fi, y="feature", x="importance", orient="h")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
