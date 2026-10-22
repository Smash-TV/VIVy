# -*- coding: utf-8 -*-
"""公开人格内核路径常量，仅依赖标准库；不含原宿主集成配置。"""

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PERSONALITY_CORE_CONFIG_PATH = os.path.join(CONFIGS_DIR, "personality_core.json")
PERSONALITY_CORE_DB_PATH = os.path.join(DATA_DIR, "personality_core.db")
