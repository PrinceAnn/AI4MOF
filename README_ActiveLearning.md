# MOF材料配比主动学习优化框架

## 概述

这个主动学习框架用于优化MOF材料的试剂配比，目标是：
1. 使5个二分类变量都为1（Crystallized, FCC_Phase, Mesoporous, Uniform_Mesoporous, Non_Spherical）
2. 最大化 Intensity_Ratio 回归值

## 工作流程

### 1. 数据预处理
```bash
cd /Users/wangzian/workspace/AI4S/AL4MOF/scripts
python data_preprocess.py
```
生成文件：
- `data/all_data_cleaned.csv` - 清洗后的数据
- `data/column_stats.csv` - 列统计信息

### 2. 训练Surrogate模型
```bash
python model.py
```
生成文件：
- `data/best_model.pth` - 训练好的DNN模型（包含5个二分类 + 1个回归任务）

### 3. 运行主动学习优化
```bash
python active_learning.py --iter 1 --rollout 100 --top_sample 20
```

参数说明：
- `--iter`: 迭代轮数（默认1）
- `--rollout`: MCTS rollout步数（默认100）
- `--top_sample`: 最终选择的样本数（默认20）

## 算法说明

### 核心算法：基于MCTS的主动学习

1. **Surrogate Model (Oracle)**
   - 输入：5个试剂用量 [HCl_mL, CH3COOH_mL, ZrCl4_mmol, HfCl4_mmol, Water_mL]
   - 输出：5个二分类概率 + 1个回归值
   - 综合得分：`score = binary_quality × normalized_intensity_ratio`
     - 如果所有二分类 > 0.5：完整得分
     - 否则：惩罚得分（×0.1）

2. **MCTS搜索策略**
   - 从得分最高的样本开始
   - 使用UCT (Upper Confidence Bound for Trees) 平衡exploration/exploitation
   - 动作空间：
     - 增加/减少单个特征
     - 随机扰动多个特征
     - 完全随机采样
   - 约束：所有特征值限制在有效范围内

3. **样本选择策略**
   - 综合考虑：
     - 预测得分高（60%权重）
     - 与历史样本距离远（多样性，40%权重）
   - 使用TSNE可视化样本分布

## 输出文件

每轮迭代在 `Results_AL/Round{N}/` 生成：

1. **history_evaluation.csv** - 历史数据的模型评估
2. **all_candidates.csv** - 所有生成的候选样本及其预测
3. **selected_samples.csv** - 最终选择的样本（用于下一轮实验）
4. **sample_selection.png** - TSNE/PCA可视化
5. **diversity_distribution.png** - 样本多样性分布
6. **feature_distributions.png** - 特征分布对比
7. **binary_predictions.png** - 二分类预测结果

## 输出格式示例

`selected_samples.csv`:
```
HCl_mL,CH3COOH_mL,ZrCl4_mmol,HfCl4_mmol,Water_mL,score,Crystallized_prob,FCC_Phase_prob,Mesoporous_prob,Uniform_Mesoporous_prob,Non_Spherical_prob,intensity_ratio_pred
0.6,0.4,0.08,0.02,3.0,0.85,0.95,0.92,0.98,0.96,0.94,3.8
...
```

## 迭代式优化流程

### Round 1: 初始探索
```bash
python active_learning.py --iter 1 --rollout 100 --top_sample 20
```
- 从初始数据集开始
- 生成20个最优候选样本

### Round 2: 基于实验结果
1. 进行实验，获得20个样本的真实结果
2. 将新数据添加到 `all_data_cleaned.csv`
3. 重新训练模型：
   ```bash
   python model.py
   ```
4. 运行第2轮优化：
   ```bash
   python active_learning.py --iter 2 --rollout 100 --top_sample 20
   ```

### Round N: 持续优化
重复上述过程直到找到满意的材料配比

## 关键参数调优

### 1. 探索权重 (weight)
```python
weight = 0.02  # 默认值
```
- 增大：更多exploration（探索未知区域）
- 减小：更多exploitation（利用已知好区域）

### 2. Rollout步数
```bash
--rollout 100  # 默认
--rollout 200  # 更彻底的搜索，但更慢
```

### 3. 起始点选择策略
```python
list1 = [5, 8, 2, 5, 1, 1]
# [运行次数, top起始点数, 随机起始点数, top得分样本数, top访问样本数, 随机样本数]
```

## 高级用法

### 自定义优化目标

修改 `oracle` 函数中的得分计算：

```python
# 当前：平衡二分类和回归
score = binary_quality * (reg_value / 5.0)

# 优先回归值：
score = binary_quality * (reg_value / 5.0) ** 2

# 严格要求二分类：
if all_binary_high and np.all(binary_probs > 0.8):
    score = reg_value / 5.0
else:
    score = 0
```

### 调整特征搜索范围

修改 `feature_ranges`:
```python
feature_ranges = {
    "HCl_mL": (0.3, 0.8),  # 缩小搜索范围
    "CH3COOH_mL": (0.2, 1.0),
    # ...
}
```

## 故障排除

### 问题1：生成候选样本太少
- 增加 `--rollout` 步数
- 增加 `list1[0]` (运行次数)
- 增大探索权重 `weight`

### 问题2：样本质量不高
- 检查模型训练质量（R² > 0.95）
- 调整起始点选择策略
- 增加更多历史数据

### 问题3：样本多样性不足
- 增大样本选择中多样性的权重（修改代码中的0.4）
- 增加随机起始点数量 `list1[2]`

## 可视化分析

运行后查看：
1. **TSNE图**：观察样本在特征空间的分布
2. **特征分布图**：对比选择样本与历史数据的差异
3. **二分类概率图**：验证二分类目标是否满足

## 下一步

完成一轮主动学习后：
1. 根据 `selected_samples.csv` 进行实验
2. 测量真实的材料性能
3. 更新数据集
4. 重新训练模型
5. 进行下一轮优化

目标：通过3-5轮迭代找到最优材料配比！
