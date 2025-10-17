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
# -*- coding: utf-8 -*-
"""
main.py
数智风控 - 多模型信用风险评估主程序
---------------------------------
统一流程：
1️⃣ 读取配置与数据
2️⃣ 执行各模型的 CV 训练与预测
3️⃣ 输出评估指标与可视化结果
4️⃣ 融合集成 (Weighted / Stacking)
"""

import os
import yaml
import pandas as pd

from common.utils import save_json, make_dir
from common.evaluation import (
    plot_label_balance,
    plot_model_comparison,
    compute_metrics,
)
from models.logistic_woe import LogisticWOEModel
from models.lightgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.extratrees_model import ExtraTreesModel
from models.ensemble_blender import EnsembleBlender


# =========================================================
# 主函数入口
# =========================================================
def main():
    # -------------------------
    # 读取配置
    # -------------------------
    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]
    target_col = cfg["data"]["target"]
    out_dir = "output"
    make_dir(out_dir)

    # -------------------------
    # 加载数据
    # -------------------------
    train_df = pd.read_excel(train_path) if train_path.endswith(".xlsx") else pd.read_csv(train_path)
    test_df = pd.read_excel(test_path) if test_path.endswith(".xlsx") else pd.read_csv(test_path)
    print(f"✅ 数据加载完成：train={train_df.shape}, test={test_df.shape}")

    # -------------------------
    # 标签分布可视化
    # -------------------------
    print("📊 绘制标签分布图...")
    plot_label_balance(train_df, path=f"{out_dir}/label_balance.png")

    # -------------------------
    # 初始化模型
    # -------------------------
    print("🚀 初始化模型...")
    lr = LogisticWOEModel(encoder_type='woe', calibrate_method='sigmoid', use_smote=True)
    lgb = LightGBMModel()
    xgb = XGBoostModel()
    cat = CatBoostModel()
    etr = ExtraTreesModel()

    # -------------------------
    # 执行各模型交叉验证
    # -------------------------
    print("🔁 开始交叉验证...")
    results = {}
    results["LogisticWOE"] = lr.run_cv(train_df, out_dir)
    results["LightGBM"] = lgb.run_cv(train_df, out_dir)
    results["XGBoost"] = xgb.run_cv(train_df, out_dir)
    results["CatBoost"] = cat.run_cv(train_df, out_dir)
    results["ExtraTrees"] = etr.run_cv(train_df, out_dir)

    # -------------------------
    # 汇总指标并可视化
    # -------------------------
    summary_df = pd.DataFrame([
        {"model": k, **v["result"]} for k, v in results.items()
    ])
    summary_path = f"{out_dir}/summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📈 模型评估结果已保存到：{summary_path}")

    plot_model_comparison(summary_df, f"{out_dir}/model_comparison.png")

    # -------------------------
    # 模型融合（集成）
    # -------------------------
    print("🤖 集成融合中...")
    weights = cfg["model"]["ensemble_weights"]
    method = cfg["model"]["blend_method"]

    oof_preds = [v["oof_cal"] for v in results.values()]
    y_true = list(results.values())[0]["y"]

    blender = EnsembleBlender(weights=weights, method=method)
    blended = blender.run(oof_preds, oof_preds, y_true=y_true)

    metrics_final = compute_metrics(y_true, blended)
    save_json(metrics_final, f"{out_dir}/metrics_blend.json")

    print("✅ 融合完成，最终指标：")
    print(metrics_final)

    print("\n🎉 全流程完成！结果已输出至 output 目录。")


# =========================================================
# 主程序执行
# =========================================================
if __name__ == "__main__":
    main()
