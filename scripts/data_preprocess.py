import pandas as pd
import os
# File paths
input_file = "/Users/wangzian/workspace/AI4S/AL4MOF/data/raw/全部数据/所有数据.xlsx"
output_file = "/Users/wangzian/workspace/AI4S/AL4MOF/data/all_data_cleaned.csv"

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