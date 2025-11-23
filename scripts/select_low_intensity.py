"""
从数据集中选择Intensity_Ratio不是NA的样本，按照Intensity_Ratio从小到大排序，取前N个

使用方法:
    python select_low_intensity.py --n_samples 35 --output low_intensity_35.csv
"""

import pandas as pd
import numpy as np
import argparse
import os

def select_low_intensity_samples(input_file, output_file, n_samples=35):
    """
    选择Intensity_Ratio最小的N个样本
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        n_samples: 要选择的样本数量
    """
    # 读取数据（保持"NA"为字符串）
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, keep_default_na=False, na_values=[''])
    total_samples = len(df)
    
    print(f"Total samples in dataset: {total_samples}")
    
    # 过滤掉Intensity_Ratio为"NA"的样本
    print("\nFiltering samples with valid Intensity_Ratio...")
    valid_mask = df['Intensity_Ratio'] != 'NA'
    valid_df = df[valid_mask].copy()
    
    # 转换Intensity_Ratio为数值
    valid_df['Intensity_Ratio'] = pd.to_numeric(valid_df['Intensity_Ratio'], errors='coerce')
    
    # 去除转换后仍然是NaN的样本（如果有的话）
    valid_df = valid_df.dropna(subset=['Intensity_Ratio'])
    
    print(f"Samples with valid Intensity_Ratio: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("Error: No valid Intensity_Ratio samples found!")
        return
    
    # 按Intensity_Ratio从小到大排序
    print(f"\nSorting by Intensity_Ratio (ascending)...")
    sorted_df = valid_df.sort_values('Intensity_Ratio', ascending=True).reset_index(drop=True)
    
    # 选择前N个
    if n_samples > len(sorted_df):
        print(f"Warning: Requested {n_samples} samples, but only {len(sorted_df)} available.")
        print(f"Using all {len(sorted_df)} samples instead.")
        n_samples = len(sorted_df)
    
    selected_df = sorted_df.head(n_samples)
    
    # 统计信息
    print("\n=== Selected Samples Statistics ===")
    print(f"Selected samples: {len(selected_df)}")
    print(f"Intensity_Ratio range: [{selected_df['Intensity_Ratio'].min():.4f}, {selected_df['Intensity_Ratio'].max():.4f}]")
    print(f"Intensity_Ratio mean: {selected_df['Intensity_Ratio'].mean():.4f}")
    print(f"Intensity_Ratio median: {selected_df['Intensity_Ratio'].median():.4f}")
    
    # 统计二分类变量
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", 
                   "Uniform_Mesoporous", "Non_Spherical"]
    print("\nBinary classifications:")
    for col in binary_cols:
        count_1 = (selected_df[col] == 1).sum()
        count_0 = (selected_df[col] == 0).sum()
        print(f"  {col}: {count_1} (1s) / {count_0} (0s)")
    
    # 统计5个二分类都为1的样本
    all_binary_1 = (selected_df[binary_cols] == 1).all(axis=1).sum()
    print(f"\nSamples with all binary=1: {all_binary_1}/{len(selected_df)}")
    
    # 保存结果
    print(f"\nSaving results to {output_file}...")
    selected_df.to_csv(output_file, index=False)
    
    print(f"✓ Done! Selected samples saved successfully.")
    
    # 显示前10个样本预览
    print("\n=== First 10 Samples (Preview) ===")
    feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]
    preview_cols = feature_cols + ['Intensity_Ratio']
    print(selected_df[preview_cols].head(10).to_string(index=True))
    
    return selected_df


def main():
    parser = argparse.ArgumentParser(
        description='Select samples with lowest Intensity_Ratio values'
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
        default='../data/low_intensity_filtered.csv',
        help='Output CSV file path (default: ../data/low_intensity_filtered.csv)'
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        default=50,
        help='Number of samples to select (default: 50)'
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
    
    # 执行选择
    select_low_intensity_samples(
        input_file=args.input,
        output_file=args.output,
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()
