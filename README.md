# 🚀 广告点击率预估 (CTR Prediction) 项目

本项目旨在通过用户画像与广告上下文数据，构建高精度的机器学习模型预测用户点击行为。代码完整复现了从大规模数据清洗、深度特征工程、Optuna 超参数自动调优到 5 折交叉验证的全流程。

**项目核心**：挖掘 `device_price` 与 `city_rank` 等特征的深层交互，解决非平衡分类问题。

---

## 📂 1. 数据集

*   **名称**: 用户行为与广告投放分析数据集
*   **来源**: [ModelScope - Dataset for User Behavior and Advertising Placement Analysis](https://modelscope.cn/datasets/aorasnli/Dataset_for_User_Behavior_and_Advertising_Placement_Analysis)
*   **规模**:
    *   训练集: 9,000,000 条记录
    *   测试集: 500,000 条记录
*   **字段说明**: 包含用户 ID (`uid`)、广告 ID (`adv_id`)、设备信息 (`device_price`, `device_name`)、应用信息 (`app_score`, `app_first_class`) 等 35+ 维度特征。

## 🛠️ 2. 技术栈

*   **Language**: Python 3
*   **Core Libraries**:
    *   `LightGBM`: 高性能 GBDT 框架。
    *   `Optuna`: 自动超参数优化框架。
    *   `Pandas/Numpy`: 数据处理与矩阵运算。
    *   `Scikit-learn`: 评估指标 (AUC) 与数据划分。

## 🧠 3. 核心算法逻辑 (The "Secret")

我深入分析了你的代码，以下是项目中最具价值的算法设计细节：

### 3.1 深度特征工程 (Feature Engineering)
你并没有直接使用原始特征，而是通过**特征交叉 (Feature Crossing)** 构建了 10 个高阶特征，这是提升模型上限的关键：

| 特征名称 | 构造逻辑 | 物理意义 |
| :--- | :--- | :--- |
| **device_city** | `device_price` × `city_rank` | **购买力 × 城市等级** (最强组合特征) |
| **app_class_score** | `app_first_class` × `app_score` | **APP热度 × 评分** (内容质量加权) |
| **online_ratio** | `communication_avgonline_30d` / `up_life_duration` | **用户活跃度比率** (防除零处理) |
| **shelf_adv** | `his_on_shelf_time` × `adv_prim_id` | **广告时长 × 广告主** (流量主偏好) |

> **特征筛选**: 代码中明智地剔除了弱相关特征（如 `age`, `gender`, `pt_d`），并利用 `nunique() <= 10` 规则自动将低基数特征转换为 `category` 类型，极大提升了 LightGBM 的训练效率。

### 3.2 模型训练策略
1.  **超参数优化**: 使用 `Optuna` 搜索了 `learning_rate`, `max_depth`, `num_leaves`, `reg_alpha/lambda` 等 8 个关键参数。
2.  **交叉验证**: 采用 **StratifiedKFold (n_splits=5)**，确保每一折的数据分布与原始数据一致。
3.  **早停机制**: 设置 `early_stopping(50)`，防止过拟合。

## 📈 4. 实验结果

经过 Optuna 调优与 5 折集成，模型在验证集上表现稳定。

*   **最终 OOF AUC**: **0.6975**
*   **单折最佳表现**: 0.69865
*   **关键发现**: 特征 `slot_id` (广告位) 和 `adv_prim_id` (广告主 ID) 是决定点击率的最强因子。

### 特征重要性 Top 5
1.  `slot_id` (广告位 ID)
2.  `adv_prim_id` (广告主 ID)
3.  `list_time` (列表时间)
4.  `career` (职业)
5.  `spread_app_id` (推广 APP ID)


## 🚀 5. 快速开始

### 5.1 环境依赖
```bash
pip install lightgbm optuna pandas scikit-learn tqdm
