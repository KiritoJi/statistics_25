# -*- coding: utf-8 -*-
"""
common/utils.py
通用工具函数集合
---------------------------------
包含：
1️⃣ save_json / load_json
2️⃣ make_dir
"""

import os
import json

def save_json(data, path):
    """保存字典为 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_dir(path):
    """确保路径存在"""
    os.makedirs(path, exist_ok=True)
