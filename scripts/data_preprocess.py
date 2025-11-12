import pandas as pd
import os
import numpy as np
os.chdir("/Users/wangzian/workspace/AI4S/AL4MOF/")
# File paths
input_file = "data/raw/全部数据/所有数据.xlsx"
output_file = "data/all_data_cleaned.csv"

# Read the Excel file
df = pd.read_excel(input_file)

# Rename columns to English
df.columns = [
    "HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL",
    "Crystallized", "FCC_Phase", "Mesoporous", "Uniform_Mesoporous", "Non_Spherical",
    "Intensity_Ratio"
]

# Map binary variables to 0 and 1
binary_mapping = {"yes": 1, "no": 0}
df["Crystallized"] = df["Crystallized"].map(binary_mapping)
df["FCC_Phase"] = df["FCC_Phase"].map({"FCC": 1, "hcp": 0, "no": 0,"hcp/FCC":0})
df["Mesoporous"] = df["Mesoporous"].map(binary_mapping)
df["Uniform_Mesoporous"] = df["Uniform_Mesoporous"].map(binary_mapping)
df["Non_Spherical"] = df["Non_Spherical"].map(binary_mapping)

# Replace "no" in the last column with "NA"
df["Intensity_Ratio"] = df["Intensity_Ratio"].replace("no", "NA")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
# Save the cleaned data to a new CSV file
df.to_csv(output_file, index=False)

print(f"Cleaned CSV file saved to {output_file}")

na_counts = df.isna().sum()
print("NA counts per column:", na_counts)

# ---------- 新增：列范围统计并保存到 CSV ----------
def _is_binary(vals):
    # 判断一列是否为二分类（只包含 0/1）
    if len(vals) == 0:
        return False
    norm = set()
    for v in vals:
        if pd.isna(v):
            continue
        # 把可转换为数值的字符串也处理
        try:
            fv = float(v)
            if fv == 0.0 or fv == 1.0:
                norm.add(int(fv))
            else:
                norm.add(fv)
        except Exception:
            norm.add(str(v))
    return norm <= {0, 1}

stats = []
for col in df.columns:
    s = df[col]
    # 把显式的 "NA" 视作缺失值进行判断
    s_clean = s.replace("NA", np.nan).dropna()
    unique_vals = pd.unique(s_clean)
    binary_flag = _is_binary(unique_vals)
    # 尝试数值化来获取 min/max（如果有数值）
    numeric = pd.to_numeric(s_clean, errors='coerce')
    if numeric.notna().any():
        num_nonnull = numeric.dropna()
        minv = num_nonnull.min()
        maxv = num_nonnull.max()
    else:
        minv = None
        maxv = None
    sample_vals = ", ".join(map(str, list(unique_vals)[:5]))
    stats.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "binary_variable": bool(binary_flag),
        "min": minv,
        "max": maxv,
        "unique_count": int(len(unique_vals)),
        "sample_values": sample_vals
    })

stats_df = pd.DataFrame(stats)
stats_out = "data/column_stats.csv"
os.makedirs(os.path.dirname(stats_out), exist_ok=True)
stats_df.to_csv(stats_out, index=False)

print(f"Column stats saved to {stats_out}")
print(stats_df.to_string(index=False))