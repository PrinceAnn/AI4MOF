"""
主动学习框架用于MOF材料配比优化

目标：
1. 5个二分类变量都为1 (Crystallized, FCC_Phase, Mesoporous, Uniform_Mesoporous, Non_Spherical)
2. 最大化 Intensity_Ratio

策略：
- 使用训练好的DNN模型作为surrogate model
- 采用基于MCTS的主动学习算法进行优化
- 通过exploration和exploitation平衡来寻找最优配比
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import argparse

# 导入模型定义
from model import HierarchicalMOFModel, MOFDataset

############################### 设置参数 ###############################

parser = argparse.ArgumentParser(description='Active Learning for MOF Optimization')
parser.add_argument('--iter', type=int, default=1, help='iteration number')
parser.add_argument('--rollout', type=int, default=100, help='number of rollout steps')
parser.add_argument('--top_sample', type=int, default=20, help='number of samples to select')
args = parser.parse_args()

# 参数设置
round_num = args.iter
rollout_round = args.rollout
top_sample = args.top_sample
round_name = f'Round{round_num}'

# 优化参数
n_dim = 5  # 5个输入特征
n_model = 1  # 使用1个模型进行预测
weight = 0.02  # UCT exploration weight
list1 = [5, 8, 2, 5, 1, 1]  # [run times, top start points, random start points, top score samples, top visit samples, random samples]

# 输入范围（从column_stats.csv获得）
feature_ranges = {
    "HCl_mL": (0.0, 1.0),
    "CH3COOH_mL": (0.0, 1.0),
    "ZrCl4_mmol": (0.0, 0.1),
    "HfCl4_mmol": (0.0, 0.1),
    "Water_mL": (1.5, 4.5)
}
feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]

# 创建结果文件夹
model_folder = "Results_AL"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(f"{model_folder}/{round_name}", exist_ok=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

############################### 加载模型和数据 ###############################

def load_model_and_data():
    """加载训练好的模型和数据集"""
    # 加载数据集
    data_path = "/Users/wangzian/workspace/AI4S/AL4MOF/data/all_data_cleaned.csv"
    dataset = MOFDataset(data_path)
    
    # 加载模型
    model_path = "/Users/wangzian/workspace/AI4S/AL4MOF/data/best_model.pth"
    # 使用weights_only=False来加载包含sklearn对象的checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = HierarchicalMOFModel(
        input_dim=5,
        hidden_dims=[64, 32, 16],
        dropout=0.2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, dataset

############################### Surrogate Model (Oracle) ###############################

def oracle(model, dataset, X, device):
    """
    使用训练好的模型预测
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象（用于标准化）
        X: 输入样本 shape=(N, 5)
        device: 计算设备
    
    Returns:
        scores: 综合得分（考虑二分类和回归）
        binary_preds: 二分类预测
        regression_preds: 回归预测
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # 标准化输入
    X_scaled = dataset.scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        binary_pred, regression_pred = model(X_tensor)
        binary_pred = binary_pred.cpu().numpy()
        regression_pred = regression_pred.cpu().numpy().flatten()
    
    # 计算综合得分
    scores = []
    for i in range(len(X)):
        # 二分类预测（0-1之间的概率）
        binary_probs = binary_pred[i]
        
        # 只有当所有二分类预测概率都较高时，才考虑回归值
        all_binary_high = np.all(binary_probs > 0.5)
        binary_quality = np.prod(binary_probs)  # 所有概率的乘积
        
        # 回归预测（反标准化）
        if dataset.scaler_y is not None:
            reg_value = dataset.scaler_y.inverse_transform([[regression_pred[i]]])[0, 0]
        else:
            reg_value = regression_pred[i]
        
        # 综合得分：二分类质量 * 回归值（归一化到0-1）
        # 如果二分类不满足，则大幅降低得分
        if all_binary_high:
            score = binary_quality * (reg_value / 5.0)  # 假设最大Intensity_Ratio约为5
        else:
            score = binary_quality * 0.1  # 惩罚不满足二分类条件的样本
        
        scores.append(score)
    
    return np.array(scores), binary_pred, regression_pred

############################### MCTS-based Active Learning Algorithm ###############################

class ActiveLearningMCTS:
    """基于MCTS的主动学习算法"""
    
    def __init__(self, exploration_weight=1.0):
        self.N = defaultdict(int)  # 访问次数
        self.children = dict()  # 子节点
        self.exploration_weight = exploration_weight
    
    def choose(self, node):
        """选择最佳后继节点"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        
        log_N_vertex = math.log(self.N[node] + 1)
        
        def uct(n):
            """UCT (Upper Confidence Bound for Trees)"""
            return n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )
        
        # 选择UCT最高的节点
        best_node = max(self.children[node], key=uct)
        
        # 随机选择一个节点用于多样性
        rand_index = random.randint(0, len(list(self.children[node])) - 1)
        rand_node = list(self.children[node])[rand_index]
        
        print(f'Visit count: {self.N[node]}, Best UCT: {uct(best_node):.4f}, Best value: {best_node.value:.4f}')
        
        if uct(best_node) > uct(node):
            return best_node, rand_node
        return node, rand_node
    
    def do_rollout(self, node):
        """执行一次rollout"""
        self._expand(node)
        self._backpropagate(node)
    
    def _expand(self, node):
        """扩展节点"""
        actions = list(range(n_dim))
        self.children[node] = node.find_children(actions)
    
    def _backpropagate(self, path):
        """反向传播更新访问次数"""
        self.N[path] += 1

# 定义优化任务节点
_OT = namedtuple("opt_task", "tup value terminal")

class OptTask(_OT):
    """优化任务节点"""
    
    def find_children(self, actions):
        """生成子节点（新的配比方案）"""
        if self.terminal:
            return set()
        
        all_tup = []
        
        for index in actions:
            tup = list(self.tup)
            flip = random.randint(0, 6)
            
            # 根据特征范围确定步长
            feature_name = feature_cols[index]
            min_val, max_val = feature_ranges[feature_name]
            step = (max_val - min_val) * 0.1  # 10%的范围作为步长
            
            if flip == 0:  # 增加
                tup[index] += step
            elif flip == 1:  # 减少
                tup[index] -= step
            elif flip == 2:  # 随机扰动多个特征（大幅度）
                for _ in range(n_dim // 2):
                    idx = random.randint(0, n_dim - 1)
                    feat_name = feature_cols[idx]
                    min_v, max_v = feature_ranges[feat_name]
                    tup[idx] = random.uniform(min_v, max_v)
            elif flip == 3:  # 随机扰动少数特征
                for _ in range(n_dim // 3):
                    idx = random.randint(0, n_dim - 1)
                    feat_name = feature_cols[idx]
                    min_v, max_v = feature_ranges[feat_name]
                    tup[idx] = random.uniform(min_v, max_v)
            elif flip == 4:  # 完全随机
                feat_name = feature_cols[index]
                min_v, max_v = feature_ranges[feat_name]
                tup[index] = random.uniform(min_v, max_v)
            else:  # 小幅度随机扰动
                noise = random.uniform(-step/2, step/2)
                tup[index] += noise
            
            # 限制在有效范围内
            for i, feat_name in enumerate(feature_cols):
                min_val, max_val = feature_ranges[feat_name]
                tup[i] = np.clip(tup[i], min_val, max_val)
            
            all_tup.append(np.array(tup))
        
        # 使用oracle评估所有生成的配比
        all_tup = np.array(all_tup)
        all_values, _, _ = oracle(global_model, global_dataset, all_tup, device)
        
        is_terminal = False
        return {OptTask(tuple(t), v, is_terminal) for t, v in zip(all_tup, all_values)}
    
    def is_terminal(self):
        """判断是否为终止节点"""
        return self.terminal
    
    def __hash__(self):
        """使节点可哈希"""
        return hash(self.tup)
    
    def __eq__(self, other):
        """节点比较"""
        return isinstance(other, OptTask) and self.tup == other.tup

############################### 辅助函数 ###############################

def most_visit_node(tree, X, top_n):
    """返回访问次数最多的节点"""
    N_visit = tree.N
    children_nodes = [node for node in tree.children]
    
    children_N = []
    X_top = []
    
    for child in children_nodes:
        child_tup = np.array(child.tup)
        # 检查是否已在X中
        same = np.all(np.abs(child_tup - X) < 1e-6, axis=1)
        has_same = any(same)
        
        if not has_same:
            children_N.append(N_visit[child])
            X_top.append(child_tup)
    
    if len(children_N) == 0:
        return np.array([])
    
    children_N = np.array(children_N)
    X_top = np.array(X_top)
    
    actual_top_n = min(top_n, len(children_N))
    ind = np.argpartition(children_N, -actual_top_n)[-actual_top_n:]
    
    return X_top[ind]

def random_node(new_x, n):
    """返回随机节点"""
    if len(new_x) == 0:
        return []
    n = min(n, len(new_x))
    indices = random.sample(range(len(new_x)), n)
    return [new_x[i] for i in indices]

############################### 单次运行 ###############################

def single_run(X, y, initial_X, initial_y, exploration_weight):
    """
    从一个初始点开始进行MCTS搜索
    
    Args:
        X: 已有的样本
        y: 已有样本的得分
        initial_X: 初始搜索点
        initial_y: 初始点的得分
        exploration_weight: 探索权重
    
    Returns:
        X_next: 选择的新样本
    """
    # 创建初始节点
    board = OptTask(tup=tuple(initial_X), value=initial_y, terminal=False)
    tree = ActiveLearningMCTS(exploration_weight=exploration_weight)
    
    boards = []
    boards_rand = []
    
    # 执行rollout
    for i in tqdm(range(rollout_round), desc="Rollout"):
        tree.do_rollout(board)
        board, board_rand = tree.choose(board)
        boards.append(list(board.tup))
        boards_rand.append(list(board_rand.tup))
    
    # 去重
    boards = np.array(boards)
    boards = np.unique(boards, axis=0)
    
    # 评估所有生成的配比
    pred_values, _, _ = oracle(global_model, global_dataset, boards, device)
    
    print(f'Generated {len(boards)} unique candidates')
    
    # 过滤掉已存在的样本
    new_x = []
    new_pred = []
    
    for i, j in zip(boards, pred_values):
        temp_x = np.array(i)
        same = np.all(np.abs(temp_x - X) < 1e-6, axis=1)
        has_same = any(same)
        
        if not has_same:
            new_pred.append(j)
            new_x.append(temp_x)
    
    new_x = np.array(new_x) if len(new_x) > 0 else np.array([]).reshape(0, n_dim)
    new_pred = np.array(new_pred)
    
    print(f'Found {len(new_x)} new candidates')
    
    if len(new_x) == 0:
        return np.array([]).reshape(0, n_dim)
    
    # 选择最优样本
    top_n = list1[3]
    actual_top_n = min(top_n, len(new_pred))
    ind = np.argpartition(new_pred, -actual_top_n)[-actual_top_n:]
    top_prediction = new_x[ind]
    
    # 选择访问最多的节点
    X_most_visit = most_visit_node(tree, X, list1[4])
    
    # 随机选择
    X_rand = random_node(new_x.tolist(), list1[5])
    X_rand = np.array(X_rand) if len(X_rand) > 0 else np.array([]).reshape(0, n_dim)
    
    # 合并
    X_next = top_prediction
    if len(X_most_visit) > 0:
        X_next = np.vstack([X_next, X_most_visit])
    if len(X_rand) > 0:
        X_next = np.vstack([X_next, X_rand])
    
    return X_next

############################### 主运行函数 ###############################

def run(X, y):
    """
    主动学习主循环
    
    Args:
        X: 当前所有样本 shape=(N, 5)
        y: 当前所有样本的得分 shape=(N,)
    
    Returns:
        top_X: 选择的新样本
    """
    # 选择起始点
    top_select = list1[1]  # 最高得分点数量
    random_select = list1[2]  # 随机点数量
    
    # 选择得分最高的点
    actual_top = min(top_select, len(y))
    ind = np.argpartition(y, -actual_top)[-actual_top:]
    
    # 随机选择一些点
    ind_random = np.setdiff1d(np.arange(len(y)), ind)
    actual_random = min(random_select, len(ind_random))
    if actual_random > 0:
        ind2 = np.random.choice(ind_random, actual_random, replace=False)
        ind = np.concatenate((ind, ind2))
    
    print(f"Starting points indices: {ind}")
    print(f"Starting points scores: {y[ind]}")
    
    x_current = X[ind]
    y_current = y[ind]
    
    # 计算exploration weight
    max_score = np.max(y)
    exploration_weight = weight * max_score if max_score > 0 else weight
    
    # 从每个起始点进行搜索
    X_all = []
    
    for i in range(len(x_current)):
        print(f"\n{'='*60}")
        print(f"Starting from point {i+1}/{len(x_current)}")
        print(f"Initial config: {x_current[i]}")
        print(f"Initial score: {y_current[i]:.4f}")
        print(f"{'='*60}")
        
        x_new = single_run(X, y, x_current[i], y_current[i], exploration_weight)
        
        if len(x_new) > 0:
            X_all.append(x_new)
    
    # 合并所有结果
    if len(X_all) > 0:
        top_X = np.vstack(X_all)
    else:
        top_X = np.array([]).reshape(0, n_dim)
    
    print(f"\nTotal candidates generated: {len(top_X)}")
    
    return top_X

############################### 样本选择（基于TSNE和得分） ###############################

def select_final_samples(all_X, sample_X, sample_scores, top_n=20):
    """
    使用TSNE可视化和多样性选择最终样本
    
    Args:
        all_X: 所有历史样本
        sample_X: 候选样本
        sample_scores: 候选样本得分
        top_n: 选择的样本数量
    
    Returns:
        selected_indices: 选择的样本索引
    """
    if len(sample_X) == 0:
        return np.array([])
    
    # 合并数据用于TSNE
    total = np.vstack([all_X, sample_X])
    
    # TSNE降维
    print("Performing TSNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(total)-1), 
                random_state=42, max_iter=1000)
    total_tsne = tsne.fit_transform(total)
    
    # 计算每个候选样本到历史样本的最小距离（多样性）
    sample_tsne = total_tsne[len(all_X):]
    history_tsne = total_tsne[:len(all_X)]
    
    sample_dist = []
    for i in range(len(sample_X)):
        distances = np.sqrt(np.sum((history_tsne - sample_tsne[i])**2, axis=1))
        min_dist = np.min(distances)
        sample_dist.append(min_dist)
    
    sample_dist = np.array(sample_dist)
    
    # 归一化距离和得分
    dist_normalized = (sample_dist - sample_dist.min()) / (sample_dist.max() - sample_dist.min() + 1e-8)
    score_normalized = (sample_scores - sample_scores.min()) / (sample_scores.max() - sample_scores.min() + 1e-8)
    
    # 综合排名：得分高 + 距离远（多样性）
    combined_rank = 0.6 * score_normalized + 0.4 * dist_normalized
    
    # 选择top-n
    actual_top_n = min(top_n, len(sample_X))
    ind = np.argpartition(combined_rank, -actual_top_n)[-actual_top_n:]
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # TSNE可视化
    plt.subplot(1, 2, 1)
    plt.scatter(history_tsne[:, 0], history_tsne[:, 1], 
                c='black', alpha=0.5, label='Historical data')
    plt.scatter(sample_tsne[:, 0], sample_tsne[:, 1], 
                c=sample_scores, cmap='viridis', alpha=0.6, label='Candidates')
    plt.scatter(sample_tsne[ind, 0], sample_tsne[ind, 1], 
                c='red', marker='*', s=200, edgecolor='black', 
                label='Selected', zorder=5)
    plt.colorbar(label='Score')
    plt.title('TSNE Visualization')
    plt.legend()
    
    # PCA可视化
    plt.subplot(1, 2, 2)
    pca = PCA(n_components=2)
    total_pca = pca.fit_transform(total)
    history_pca = total_pca[:len(all_X)]
    sample_pca = total_pca[len(all_X):]
    
    plt.scatter(history_pca[:, 0], history_pca[:, 1], 
                c='black', alpha=0.5, label='Historical data')
    plt.scatter(sample_pca[:, 0], sample_pca[:, 1], 
                c=sample_scores, cmap='viridis', alpha=0.6, label='Candidates')
    plt.scatter(sample_pca[ind, 0], sample_pca[ind, 1], 
                c='red', marker='*', s=200, edgecolor='black', 
                label='Selected', zorder=5)
    plt.colorbar(label='Score')
    plt.title('PCA Visualization')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_folder}/{round_name}/sample_selection.png', dpi=300)
    plt.close()
    
    # 距离分布
    plt.figure(figsize=(8, 5))
    plt.hist(sample_dist, bins=30, color='blue', alpha=0.6, 
             edgecolor='black', label='All candidates')
    plt.hist(sample_dist[ind], bins=15, color='red', alpha=0.7, 
             edgecolor='black', label='Selected')
    plt.xlabel('Distance to nearest historical sample')
    plt.ylabel('Frequency')
    plt.title('Sample Diversity Distribution')
    plt.legend()
    plt.savefig(f'{model_folder}/{round_name}/diversity_distribution.png', dpi=300)
    plt.close()
    
    return ind

############################### 主程序 ###############################

def main():
    global global_model, global_dataset
    
    print("="*80)
    print(f"Active Learning Round {round_num}")
    print("="*80)
    
    # 加载模型和数据
    print("\nLoading model and dataset...")
    model, dataset = load_model_and_data()
    global_model = model
    global_dataset = dataset
    
    # 加载历史数据
    data = pd.read_csv("/Users/wangzian/workspace/AI4S/AL4MOF/data/all_data_cleaned.csv")
    X_history = data[feature_cols].values
    
    # 计算历史数据的得分
    print("\nEvaluating historical data...")
    y_history, binary_preds, regression_preds = oracle(model, dataset, X_history, device)
    
    print(f"\nHistorical data statistics:")
    print(f"  Total samples: {len(X_history)}")
    print(f"  Score range: [{y_history.min():.4f}, {y_history.max():.4f}]")
    print(f"  Mean score: {y_history.mean():.4f}")
    
    # 保存历史数据评估结果
    history_results = pd.DataFrame(X_history, columns=feature_cols)
    history_results['score'] = y_history
    for i, col in enumerate(["Crystallized", "FCC_Phase", "Mesoporous", 
                              "Uniform_Mesoporous", "Non_Spherical"]):
        history_results[f'{col}_prob'] = binary_preds[:, i]
    history_results['intensity_ratio_pred'] = regression_preds
    history_results.to_csv(f'{model_folder}/{round_name}/history_evaluation.csv', index=False)
    
    # 运行主动学习
    print("\n" + "="*80)
    print("Starting Active Learning Optimization")
    print("="*80)
    
    all_candidates = []
    
    for run_idx in range(list1[0]):
        print(f"\n{'#'*80}")
        print(f"# Run {run_idx + 1} / {list1[0]}")
        print(f"{'#'*80}")
        
        candidates = run(X_history, y_history)
        
        if len(candidates) > 0:
            all_candidates.append(candidates)
            print(f"Generated {len(candidates)} candidates in this run")
    
    # 合并所有候选样本
    if len(all_candidates) > 0:
        all_candidates = np.vstack(all_candidates)
        # 去重
        all_candidates = np.unique(all_candidates, axis=0)
        print(f"\nTotal unique candidates: {len(all_candidates)}")
    else:
        print("\nNo candidates generated!")
        return
    
    # 评估所有候选样本
    print("\nEvaluating all candidates...")
    candidate_scores, candidate_binary, candidate_regression = oracle(
        model, dataset, all_candidates, device
    )
    
    print(f"Candidate score range: [{candidate_scores.min():.4f}, {candidate_scores.max():.4f}]")
    print(f"Candidate mean score: {candidate_scores.mean():.4f}")
    
    # 选择最终样本
    print(f"\nSelecting top {top_sample} samples...")
    selected_ind = select_final_samples(X_history, all_candidates, 
                                        candidate_scores, top_n=top_sample)
    
    selected_samples = all_candidates[selected_ind]
    selected_scores = candidate_scores[selected_ind]
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存所有候选样本
    candidates_df = pd.DataFrame(all_candidates, columns=feature_cols)
    candidates_df['score'] = candidate_scores
    for i, col in enumerate(["Crystallized", "FCC_Phase", "Mesoporous", 
                              "Uniform_Mesoporous", "Non_Spherical"]):
        candidates_df[f'{col}_prob'] = candidate_binary[:, i]
    candidates_df['intensity_ratio_pred'] = candidate_regression
    candidates_df.to_csv(f'{model_folder}/{round_name}/all_candidates.csv', index=False)
    
    # 保存选择的样本
    selected_df = pd.DataFrame(selected_samples, columns=feature_cols)
    selected_df['score'] = selected_scores
    for i, col in enumerate(["Crystallized", "FCC_Phase", "Mesoporous", 
                              "Uniform_Mesoporous", "Non_Spherical"]):
        selected_df[f'{col}_prob'] = candidate_binary[selected_ind, i]
    selected_df['intensity_ratio_pred'] = candidate_regression[selected_ind]
    selected_df.to_csv(f'{model_folder}/{round_name}/selected_samples.csv', index=False)
    
    # 打印选择的样本
    print("\n" + "="*80)
    print("Selected Samples for Next Experiment")
    print("="*80)
    print(selected_df.to_string(index=False))
    
    # 可视化最优样本
    visualize_results(X_history, y_history, selected_samples, selected_scores, 
                     candidate_binary[selected_ind], candidate_regression[selected_ind])
    
    print(f"\n{'='*80}")
    print(f"Results saved to {model_folder}/{round_name}/")
    print(f"{'='*80}")

def visualize_results(X_history, y_history, selected_X, selected_scores, 
                     selected_binary, selected_regression):
    """可视化优化结果"""
    
    # 特征分布对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(X_history[:, i], bins=20, alpha=0.5, label='Historical', color='blue')
        ax.hist(selected_X[:, i], bins=10, alpha=0.7, label='Selected', color='red')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} Distribution')
        ax.legend()
    
    # 得分对比
    ax = axes[5]
    ax.hist(y_history, bins=20, alpha=0.5, label='Historical', color='blue')
    ax.hist(selected_scores, bins=10, alpha=0.7, label='Selected', color='red')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_folder}/{round_name}/feature_distributions.png', dpi=300)
    plt.close()
    
    # 二分类预测可视化
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", 
                   "Uniform_Mesoporous", "Non_Spherical"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, col) in enumerate(zip(axes, binary_cols)):
        ax.bar(['Selected\nSamples'], [selected_binary[:, i].mean()], 
               color='green', alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Average Probability')
        ax.set_title(col)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_folder}/{round_name}/binary_predictions.png', dpi=300)
    plt.close()
    
    print("\nVisualization saved!")

if __name__ == "__main__":
    main()
