"""
仅回归任务的MOF模型 - 用于预测Intensity_Ratio

与原模型的区别：
- 移除了二分类任务
- 只预测Intensity_Ratio回归值
- 适用于所有样本都有有效Intensity_Ratio的数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import os
os.chdir("/Users/wangzian/workspace/AI4S/AI4MOF")

# 设置随机种子
torch.manual_seed(42231)
np.random.seed(42231)

class MOFRegressionDataset(Dataset):
    """MOF回归数据集类 - 仅用于Intensity_Ratio预测"""
    def __init__(self, csv_path):
        # 读取数据（保持"NA"为字符串，虽然我们期望没有NA）
        self.data = pd.read_csv(csv_path, keep_default_na=False, na_values=[''])
        
        # 输入特征 (5个连续变量)
        self.feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]
        # 回归目标
        self.regression_col = "Intensity_Ratio"
        
        # 提取特征
        self.X = self.data[self.feature_cols].values.astype(np.float32)
        
        # 提取回归标签并转换为数值
        intensity_ratio = self.data[self.regression_col]
        self.y_regression = pd.to_numeric(intensity_ratio, errors='coerce').values.astype(np.float32)
        
        # 检查是否有NA值
        na_count = np.isnan(self.y_regression).sum()
        if na_count > 0:
            print(f"Warning: Found {na_count} NA values in Intensity_Ratio. These will be removed.")
            # 移除含有NA的样本
            valid_mask = ~np.isnan(self.y_regression)
            self.X = self.X[valid_mask]
            self.y_regression = self.y_regression[valid_mask]
            print(f"Remaining samples: {len(self.X)}")
        
        # 最大绝对值归一化输入特征
        self.scaler_X = MaxAbsScaler()
        self.X = self.scaler_X.fit_transform(self.X).astype(np.float32)
        
        # 最大绝对值归一化回归目标
        self.scaler_y = MaxAbsScaler()
        self.y_regression_scaled = self.scaler_y.fit_transform(
            self.y_regression.reshape(-1, 1)
        ).flatten().astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.X[idx], dtype=torch.float32),
            'regression_target': torch.tensor(self.y_regression_scaled[idx], dtype=torch.float32)
        }


class RegressionOnlyModel(nn.Module):
    """仅回归的MOF预测模型
    
    结构：
    - 多层全连接网络
    - 只输出Intensity_Ratio预测值
    """
    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16], dropout=0.2):
        super(RegressionOnlyModel, self).__init__()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 添加最后的回归输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            regression_target = batch['regression_target'].to(device)
            
            optimizer.zero_grad()
            regression_pred = model(features).squeeze()
            
            loss = criterion(regression_pred, regression_target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                regression_target = batch['regression_target'].to(device)
                
                regression_pred = model(features).squeeze()
                loss = criterion(regression_pred, regression_target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader, dataset, device):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            regression_target = batch['regression_target']
            
            regression_pred = model(features).squeeze()
            
            all_preds.append(regression_pred.cpu())
            all_targets.append(regression_target)
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 反标准化
    preds_original = dataset.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_original = dataset.scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # 计算指标
    mse = ((preds_original - targets_original) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(preds_original - targets_original).mean()
    
    # R² score
    ss_res = np.sum((targets_original - preds_original) ** 2)
    ss_tot = np.sum((targets_original - targets_original.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n=== Regression Results ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def predict(model, dataset, input_dict, device):
    """对单个样本进行预测
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象（用于标准化和反标准化）
        input_dict: 输入字典，如 {"HCl_mL": 0.5, "CH3COOH_mL": 0.5, ...}
        device: 计算设备
    
    Returns:
        float: 预测的Intensity_Ratio值
    """
    model.eval()
    
    # 构造输入
    feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]
    input_array = np.array([[input_dict[col] for col in feature_cols]], dtype=np.float32)
    
    # 标准化
    input_scaled = dataset.scaler_X.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        regression_pred = model(input_tensor).squeeze()
        regression_pred_scaled = regression_pred.cpu().numpy()
    
    # 反标准化
    intensity_value = dataset.scaler_y.inverse_transform([[regression_pred_scaled]])[0, 0]
    
    return float(intensity_value)


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data_path = "data/low_intensity_filtered.csv"
    dataset = MOFRegressionDataset(data_path)
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    
    # 划分数据集 (80% 训练, 10% 验证, 10% 测试)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 创建模型
    model = RegressionOnlyModel(
        input_dim=5,
        hidden_dims=[64, 32, 16],
        dropout=0.2
    ).to(device)
    
    print("\n=== Model Architecture ===")
    print(model)
    
    # 训练模型
    print("\n=== Training Model ===")
    model = train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001)
    
    # 评估模型
    print("\n=== Evaluating on Test Set ===")
    metrics = evaluate_model(model, test_loader, dataset, device)
    
    # 保存模型
    model_save_path = "data/best_model_regression_only.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': dataset.scaler_X,
        'scaler_y': dataset.scaler_y,
        'metrics': metrics
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # 示例预测
    print("\n=== Example Prediction ===")
    example_input = {
        "HCl_mL": 0.4,
        "CH3COOH_mL": 0.6,
        "ZrCl4_mmol": 0.0,
        "HfCl4_mmol": 0.1,
        "Water_mL": 3.1
    }
    result = predict(model, dataset, example_input, device)
    print(f"Input: {example_input}")
    print(f"Predicted Intensity_Ratio: {result:.4f}")


if __name__ == "__main__":
    main()
