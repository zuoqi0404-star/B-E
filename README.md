# B + E：全量 B 样本与学生/门控联合训练

入口为 `BE_model.py`。这是独立推荐模型，使用 LLM4CDSR 的预处理数据，不是 LLM4CDSR 的复现。

## 流程

1. 使用扩容 B 的 sport 训练目标训练教师，按验证集 MRR 保存最佳教师。
2. 教师对每个训练历史中的 cloth 连续兴趣片段逐个进行硬访问屏蔽，比较真实目标在全部 sport 候选中的排名。
3. 监督概率为 `sigmoid(log(r_masked / r_full) / temperature)`。屏蔽后排名变差表示片段有益；变好表示可能干扰。
4. 冻结教师，从教师复制独立学生；新建门控，从头联合训练学生与门控，不载入已训练 B 门控。
5. 损失默认为 `BCE(rank_target) + 0.1 * CE(student_prediction, next_sport_item)`。门控接收两项梯度，学生接收推荐损失梯度。教师提供固定的特征与反事实监督。
6. 按验证集 MRR（相同时按门控概率 MAE）选择学生与门控的同轮参数，之后评估测试集。

沿用 B 样本模式，不进行 C 的教师/门控用户隔离，也不使用 D 的类别重加权。在当前 seed=42 数据上，教师 4,057 条，联合阶段 2,836 条，验证和测试各 365 条。外层训练、验证、测试用户仍隔离。

## 门值与片段

`g = 2 * sigmoid(z)`，有效 cloth 片段的门值位于 (0,2)，中性参照为 1。sport 权重固定为 1。这个范围允许相对增强源域访问，但不是直接把物品打分乘以门值；权重作用于注意力访问。

片段在最多 50 个混合历史事件内动态更新，片段必须连续且属于同一领域；领域切换强制结束片段，不能跳过中间事件后连接同域物品。训练目标不进入历史。LLM 物品向量始终冻结；学生的投影、序列编码器等参数可更新。

## 数据准备

将已有且有权使用的三个文件放入项目 `handled/`：

- `cloth_sport.pkl`
- `itm_emb_np_all.pkl`
- `id_map.json`

数据、权重和用户级运行结果不上传。只加载可信的 pickle/checkpoint 文件。

## 安装与运行

使用 Python 3.12。在正常 Python 环境安装依赖：

```powershell
python -m pip install -r requirements.txt
```

需要匹配本项目已验证 CUDA 环境时，再安装 GPU 版本：

```powershell
python -m pip install -r requirements-gpu.txt
```

完整训练：

```powershell
python BE_model.py train --device cuda --teacher-epochs 100 --gate-epochs 50 --window-size 50 --seed 42 --output-dir handled/BE_seed42_100_50
```

CPU 可改为 `--device cpu`。输出目录必须不存在，防止覆盖实验。新建输出目录可以重复试验。

已有新用户历史推理（不需要真实下一个物品）：

```powershell
python BE_model.py infer --device cuda --history-json history.json --output handled/prediction.json
```

`history.json` 示例为 `{"local_item_ids": [1, 2, 3], "domain_ids": [0, 1, 1]}`，0 表示 cloth、1 表示 sport；物品编号是各域内从 1 开始的本地编号，需要与自己的数据匹配。使用本地已有用户示例可省略 `--history-json`，默认 user-id=746。

单元测试：

```powershell
python -m unittest discover -p "test_*.py"
```

## 文件与产物

- `BE_model.py`：新流程入口、联合训练约束验证、保存、推理和本地旧实验对照。
- `counterfactual_variants.py`：中性门值为 1 的门控、联合优化和评估。
- `cloth_sport_counterfactual_gate.py`：推荐器、硬访问屏蔽、排名监督和样本构建。
- `teacher_expansion_A.py`：扩容 sport 教师样本；BE 仅调用其中的样本构建函数。
- `dynamic_interest_segments.py` / `interest_segments.py`：连续同域动态片段和数据加载工具。
- `test_*.py`：片段、屏蔽、排名标签、教师样本范围、梯度与冻结等检查。

输出目录包含 `teacher.pt`、`gate.pt`（固定教师、独立学生、门控与预处理参数）、`summary.json`、标签、逐用户排名和 `COMPARISON.md`。本地若保留旧 expanded A/B/E 输出，会只读回放同一测试集并比较；全新项目不要求这些旧输出。

指标为全部 sport 候选排名的 HR@10、NDCG@10、MRR。单次结果无法说明统计稳定提升，建议预先固定评估流程并运行多个种子。与公开 LLM4CDSR 数字比较前必须统一数据划分和候选集。
