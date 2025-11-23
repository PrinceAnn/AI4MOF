"""
从剩余数据集中找到与selected_samples最接近的配方

使用方法:
    python find_closest_samples.py --selected Results_AL/Round1/selected_samples.csv --output closest_samples.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_samples(all_data_path, subset_path, selected_path, output_path, use_cosine=True):
    """
    从(all_data - subset)中找到与selected_samples最接近的配方
    
    Args:
        all_data_path: 完整数据集路径
        subset_path: 已使用的50样本子集路径
        selected_path: 主动学习选择的样本路径
        output_path: 输出文件路径
        use_cosine: 是否使用余弦相似度（True）或欧氏距离（False）
    """
    # 读取数据（保持"NA"为字符串）
    print("Reading data files...")
    all_data = pd.read_csv(all_data_path, keep_default_na=False, na_values=[''])
    subset_data = pd.read_csv(subset_path, keep_default_na=False, na_values=[''])
    selected_data = pd.read_csv(selected_path)
    
    print(f"Total samples in all_data: {len(all_data)}")
    print(f"Samples in subset (used): {len(subset_data)}")
    print(f"Selected samples to match: {len(selected_data)}")
    
    # 特征列（配方参数）
    feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]
    
    # 获取剩余数据：从all_data中移除subset中的样本
    # 使用配方参数进行精确匹配
    print("\nFinding remaining samples (all_data - subset)...")
    
    # 创建用于比较的key
    def create_key(df):
        return df[feature_cols].apply(lambda row: tuple(row.round(6)), axis=1)
    
    all_keys = create_key(all_data)
    subset_keys = create_key(subset_data)
    
    # 找到不在subset中的样本
    remaining_mask = ~all_keys.isin(subset_keys)
    remaining_data = all_data[remaining_mask].copy()
    
    print(f"Remaining samples: {len(remaining_data)}")
    
    if len(remaining_data) == 0:
        print("Error: No remaining samples found!")
        return
    
    # 提取特征
    X_remaining = remaining_data[feature_cols].values
    X_selected = selected_data[feature_cols].values
    
    # 标准化特征以计算相似度/距离
    print(f"\nStandardizing features for {'cosine similarity' if use_cosine else 'distance'} calculation...")
    scaler = StandardScaler()
    X_remaining_scaled = scaler.fit_transform(X_remaining)
    X_selected_scaled = scaler.transform(X_selected)
    
    # 为每个selected sample找到最近的remaining sample
    metric_name = "cosine similarity" if use_cosine else "euclidean distance"
    print(f"\nFinding closest matches using {metric_name}...")
    closest_indices = []
    similarities_or_distances = []
    
    if use_cosine:
        # 使用余弦相似度
        for i, selected_point in enumerate(X_selected_scaled):
            # 计算余弦相似度 (值越大越相似)
            similarities = cosine_similarity(
                selected_point.reshape(1, -1), 
                X_remaining_scaled
            ).flatten()
            closest_idx = np.argmax(similarities)  # 找最大相似度
            max_similarity = similarities[closest_idx]
            
            closest_indices.append(closest_idx)
            similarities_or_distances.append(max_similarity)
            
            print(f"  Sample {i+1}/{len(X_selected)}: similarity = {max_similarity:.4f}")
    else:
        # 使用欧氏距离
        for i, selected_point in enumerate(X_selected_scaled):
            # 计算欧氏距离 (值越小越近)
            dists = np.sqrt(np.sum((X_remaining_scaled - selected_point)**2, axis=1))
            closest_idx = np.argmin(dists)
            min_dist = dists[closest_idx]
            
            closest_indices.append(closest_idx)
            similarities_or_distances.append(min_dist)
            
            print(f"  Sample {i+1}/{len(X_selected)}: distance = {min_dist:.4f}")
    
    # 获取最接近的样本（可能有重复）
    closest_samples = remaining_data.iloc[closest_indices].copy()
    
    # 添加相似度/距离信息和对应的selected sample信息
    metric_col_name = 'cosine_similarity' if use_cosine else 'euclidean_distance'
    closest_samples[metric_col_name] = similarities_or_distances
    closest_samples['selected_sample_index'] = range(len(selected_data))
    
    # 添加selected sample的配方信息（用于对比）
    for col in feature_cols:
        closest_samples[f'selected_{col}'] = selected_data[col].values
    
    # 重新排列列顺序，方便查看
    output_cols = (
        feature_cols + 
        ['Crystallized', 'FCC_Phase', 'Mesoporous', 'Uniform_Mesoporous', 'Non_Spherical', 'Intensity_Ratio'] +
        [metric_col_name, 'selected_sample_index'] +
        [f'selected_{col}' for col in feature_cols]
    )
    
    closest_samples = closest_samples[output_cols]
    
    # 保存结果
    print(f"\nSaving results to {output_path}...")
    closest_samples.to_csv(output_path, index=False)
    
    # 统计信息
    print("\n=== Closest Samples Statistics ===")
    print(f"Total matches found: {len(closest_samples)}")
    print(f"Unique samples: {len(closest_samples.drop_duplicates(subset=feature_cols))}")
    
    if use_cosine:
        print(f"Average cosine similarity: {np.mean(similarities_or_distances):.4f}")
        print(f"Min similarity: {np.min(similarities_or_distances):.4f}")
        print(f"Max similarity: {np.max(similarities_or_distances):.4f}")
    else:
        print(f"Average distance: {np.mean(similarities_or_distances):.4f}")
        print(f"Min distance: {np.min(similarities_or_distances):.4f}")
        print(f"Max distance: {np.max(similarities_or_distances):.4f}")
    
    # 统计二分类变量
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", 
                   "Uniform_Mesoporous", "Non_Spherical"]
    print("\nBinary classifications in closest samples:")
    for col in binary_cols:
        count_1 = (closest_samples[col] == 1).sum()
        count_0 = (closest_samples[col] == 0).sum()
        print(f"  {col}: {count_1} (1s) / {count_0} (0s)")
    
    # 统计有效的Intensity_Ratio
    all_binary_1 = (closest_samples[binary_cols] == 1).all(axis=1)
    valid_intensity = (closest_samples[all_binary_1]['Intensity_Ratio'] != 'NA')
    n_valid = valid_intensity.sum()
    print(f"\nSamples with all binary=1 and valid Intensity_Ratio: {n_valid}/{len(closest_samples)}")
    
    if n_valid > 0:
        intensity_values = pd.to_numeric(
            closest_samples[all_binary_1]['Intensity_Ratio'].replace('NA', np.nan), 
            errors='coerce'
        ).dropna()
        if len(intensity_values) > 0:
            print(f"Intensity_Ratio range: [{intensity_values.min():.4f}, {intensity_values.max():.4f}]")
            print(f"Intensity_Ratio mean: {intensity_values.mean():.4f}")
    
    print(f"\n✓ Done! Results saved to {output_path}")
    
    # 显示前几个匹配
    print("\n=== First 5 Matches (Preview) ===")
    preview_cols = feature_cols + ['Intensity_Ratio', metric_col_name]
    print(closest_samples[preview_cols].head().to_string(index=False))
    
    return closest_samples


def main():
    parser = argparse.ArgumentParser(
        description='Find closest samples from remaining data to selected samples'
    )
    parser.add_argument(
        '--all_data', 
        type=str, 
        default='../data/all_data_cleaned.csv',
        help='Path to complete dataset (default: ../data/all_data_cleaned.csv)'
    )
    parser.add_argument(
        '--subset', 
        type=str, 
        default='../data/subset_50.csv',
        help='Path to 50-sample subset (default: ../data/subset_50.csv)'
    )
    parser.add_argument(
        '--selected', 
        type=str, 
        default='../Results_AL/Round1/selected_samples.csv',
        help='Path to selected samples from active learning (default: ../Results_AL/Round1/selected_samples.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='../Results_AL/Round1/closest_samples.csv',
        help='Output file path (default: ../Results_AL/Round1/closest_samples.csv)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='Similarity metric to use: cosine or euclidean (default: cosine)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for file_path, name in [
        (args.all_data, 'all_data'),
        (args.subset, 'subset'),
        (args.selected, 'selected')
    ]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found: {file_path}")
            return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行查找
    find_closest_samples(
        all_data_path=args.all_data,
        subset_path=args.subset,
        selected_path=args.selected,
        output_path=args.output,
        use_cosine=(args.metric == 'cosine')
    )


if __name__ == "__main__":
    main()
