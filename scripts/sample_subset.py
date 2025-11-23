"""
从all_data_cleaned.csv中随机选择N个样本创建子集

使用方法:
    python sample_subset.py --n_samples 50 --output subset_50.csv --seed 42
"""

import pandas as pd
import numpy as np
import argparse
import os

def sample_subset(input_file, output_file, n_samples=50, random_seed=42):
    """
    从数据集中随机选择n_samples个样本
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        n_samples: 要选择的样本数量
        random_seed: 随机种子，保证可重复性
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 读取数据 (保持"NA"字符串不被解析为空值)
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, keep_default_na=False, na_values=[''])
    total_samples = len(df)
    
    print(f"Total samples in dataset: {total_samples}")
    
    # 检查样本数量是否合理
    if n_samples > total_samples:
        print(f"Warning: Requested {n_samples} samples, but only {total_samples} available.")
        print(f"Using all {total_samples} samples instead.")
        n_samples = total_samples
        subset_df = df
    else:
        # 随机选择样本
        print(f"Randomly sampling {n_samples} samples (seed={random_seed})...")
        subset_df = df.sample(n=n_samples, random_state=random_seed)
    
    # 统计子集信息
    print("\n=== Subset Statistics ===")
    print(f"Selected samples: {len(subset_df)}")
    
    # 统计二分类变量
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", 
                   "Uniform_Mesoporous", "Non_Spherical"]
    for col in binary_cols:
        count_1 = (subset_df[col] == 1).sum()
        count_0 = (subset_df[col] == 0).sum()
        print(f"{col}: {count_1} (1s) / {count_0} (0s)")
    
    # 统计有效的Intensity_Ratio样本（5个二分类都为1）
    all_binary_1 = (subset_df[binary_cols] == 1).all(axis=1)
    valid_intensity = subset_df[all_binary_1]['Intensity_Ratio'] != 'NA'
    n_valid = valid_intensity.sum()
    print(f"\nSamples with all binary=1 and valid Intensity_Ratio: {n_valid}")
    
    if n_valid > 0:
        intensity_values = pd.to_numeric(
            subset_df[all_binary_1]['Intensity_Ratio'].replace('NA', np.nan), 
            errors='coerce'
        ).dropna()
        print(f"Intensity_Ratio range: [{intensity_values.min():.4f}, {intensity_values.max():.4f}]")
        print(f"Intensity_Ratio mean: {intensity_values.mean():.4f}")
    
    # 保存子集
    print(f"\nSaving subset to {output_file}...")
    subset_df.to_csv(output_file, index=False)
    
    print(f"✓ Done! Subset saved successfully.")
    
    return subset_df


def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample a subset from all_data_cleaned.csv'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='../data/all_data_cleaned.csv',
        help='Input CSV file path (default: ../data/all_data_cleaned.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='../data/subset_50.csv',
        help='Output CSV file path (default: ../data/subset_50.csv)'
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        default=60,
        help='Number of samples to select (default: 60)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行采样
    sample_subset(
        input_file=args.input,
        output_file=args.output,
        n_samples=args.n_samples,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
