import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import os
import argparse
os.chdir("/Users/wangzian/workspace/AI4S/AI4MOF")

class MOFDataset(Dataset):
    """MOF数据集类"""
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        # 输入特征 (5个连续变量)
        self.feature_cols = ["HCl_mL", "CH3COOH_mL", "ZrCl4_mmol", "HfCl4_mmol", "Water_mL"]
        # 二分类目标 (5个)
        self.binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", "Uniform_Mesoporous", "Non_Spherical"]
        # 回归目标
        self.regression_col = "Intensity_Ratio"
        
        # 提取特征
        self.X = self.data[self.feature_cols].values.astype(np.float32)
        
        # 提取二分类标签
        self.y_binary = self.data[self.binary_cols].values.astype(np.float32)
        
        # 提取回归标签 (将NA转换为0，但会用mask标记)
        intensity_ratio = self.data[self.regression_col].replace("NA", np.nan)
        self.y_regression = pd.to_numeric(intensity_ratio, errors='coerce').fillna(0).values.astype(np.float32)
        
        # 创建mask: 只有5个二分类都为1时才有效
        self.regression_mask = (self.y_binary.sum(axis=1) == 5).astype(np.float32)
        
        # 最大绝对值归一化输入特征
        self.scaler_X = MaxAbsScaler()
        self.X = self.scaler_X.fit_transform(self.X).astype(np.float32)
        
        # 最大绝对值归一化回归目标 (只对有效值进行归一化)
        valid_regression = self.y_regression[self.regression_mask == 1]
        if len(valid_regression) > 0:
            self.scaler_y = MaxAbsScaler()
            self.scaler_y.fit(valid_regression.reshape(-1, 1))
            self.y_regression_scaled = np.zeros_like(self.y_regression)
            self.y_regression_scaled[self.regression_mask == 1] = self.scaler_y.transform(
                self.y_regression[self.regression_mask == 1].reshape(-1, 1)
            ).flatten()
        else:
            self.scaler_y = None
            self.y_regression_scaled = self.y_regression
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.X[idx], dtype=torch.float32),
            'binary_targets': torch.tensor(self.y_binary[idx], dtype=torch.float32),
            'regression_target': torch.tensor(self.y_regression_scaled[idx], dtype=torch.float32),
            'regression_mask': torch.tensor(self.regression_mask[idx], dtype=torch.float32)
        }


class HierarchicalMOFModel(nn.Module):
    """层级MOF预测模型
    
    结构：
    1. 共享特征提取层
    2. 二分类分支：预测5个二分类变量
    3. 回归分支：预测Intensity_Ratio (仅当5个二分类都为1时有意义)
    """
    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16], dropout=0.2):
        super(HierarchicalMOFModel, self).__init__()
        
        # 共享特征提取器
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
        self.shared_encoder = nn.Sequential(*layers)
        
        # 二分类分支 (5个输出)
        self.binary_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 5),  # 5个二分类输出
            nn.Sigmoid()
        )
        
        # 回归分支 (1个输出)
        self.regression_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 回归输出
        )
    
    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_encoder(x)
        
        # 二分类预测
        binary_output = self.binary_branch(shared_features)
        
        # 回归预测
        regression_output = self.regression_branch(shared_features)
        
        return binary_output, regression_output


class HierarchicalLoss(nn.Module):
    """层级损失函数
    
    结合二分类损失和回归损失，回归损失只在5个二分类都为1时计算
    """
    def __init__(self, binary_weight=1.0, regression_weight=1.0):
        super(HierarchicalLoss, self).__init__()
        self.binary_weight = binary_weight
        self.regression_weight = regression_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, binary_pred, regression_pred, binary_target, regression_target, regression_mask):
        # 二分类损失 (所有样本)
        binary_loss = self.bce_loss(binary_pred, binary_target)
        
        # 回归损失 (仅mask为1的样本)
        regression_loss_all = self.mse_loss(regression_pred.squeeze(), regression_target)
        masked_regression_loss = (regression_loss_all * regression_mask).sum() / (regression_mask.sum() + 1e-8)
        
        # 总损失
        total_loss = self.binary_weight * binary_loss + self.regression_weight * masked_regression_loss
        
        return total_loss, binary_loss, masked_regression_loss


def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """训练模型"""
    criterion = HierarchicalLoss(binary_weight=1.0, regression_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_binary_loss = 0.0
        train_regression_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            binary_targets = batch['binary_targets'].to(device)
            regression_target = batch['regression_target'].to(device)
            regression_mask = batch['regression_mask'].to(device)
            
            optimizer.zero_grad()
            binary_pred, regression_pred = model(features)
            
            loss, b_loss, r_loss = criterion(
                binary_pred, regression_pred, 
                binary_targets, regression_target, regression_mask
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_binary_loss += b_loss.item()
            train_regression_loss += r_loss.item()
        
        train_loss /= len(train_loader)
        train_binary_loss /= len(train_loader)
        train_regression_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_binary_loss = 0.0
        val_regression_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                binary_targets = batch['binary_targets'].to(device)
                regression_target = batch['regression_target'].to(device)
                regression_mask = batch['regression_mask'].to(device)
                
                binary_pred, regression_pred = model(features)
                
                loss, b_loss, r_loss = criterion(
                    binary_pred, regression_pred, 
                    binary_targets, regression_target, regression_mask
                )
                
                val_loss += loss.item()
                val_binary_loss += b_loss.item()
                val_regression_loss += r_loss.item()
        
        val_loss /= len(val_loader)
        val_binary_loss /= len(val_loader)
        val_regression_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train - Total: {train_loss:.4f}, Binary: {train_binary_loss:.4f}, Regression: {train_regression_loss:.4f}")
            print(f"  Val   - Total: {val_loss:.4f}, Binary: {val_binary_loss:.4f}, Regression: {val_regression_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader, dataset, device):
    """评估模型性能"""
    model.eval()
    
    all_binary_preds = []
    all_binary_targets = []
    all_regression_preds = []
    all_regression_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            binary_targets = batch['binary_targets']
            regression_target = batch['regression_target']
            regression_mask = batch['regression_mask']
            
            binary_pred, regression_pred = model(features)
            
            all_binary_preds.append(binary_pred.cpu())
            all_binary_targets.append(binary_targets)
            all_regression_preds.append(regression_pred.cpu())
            all_regression_targets.append(regression_target)
            all_masks.append(regression_mask)
    
    binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
    binary_targets = torch.cat(all_binary_targets, dim=0).numpy()
    regression_preds = torch.cat(all_regression_preds, dim=0).numpy().flatten()
    regression_targets = torch.cat(all_regression_targets, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()
    
    # 二分类指标
    binary_preds_discrete = (binary_preds > 0.5).astype(int)
    binary_accuracy = (binary_preds_discrete == binary_targets).mean()
    
    # 每个二分类任务的准确率
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", "Uniform_Mesoporous", "Non_Spherical"]
    print("\n=== Binary Classification Results ===")
    for i, col in enumerate(binary_cols):
        acc = (binary_preds_discrete[:, i] == binary_targets[:, i]).mean()
        print(f"{col}: {acc:.4f}")
    print(f"Overall Binary Accuracy: {binary_accuracy:.4f}")
    
    # 回归指标 (仅对有效样本)
    valid_indices = masks == 1
    if valid_indices.sum() > 0:
        valid_preds = regression_preds[valid_indices]
        valid_targets = regression_targets[valid_indices]
        
        # 反标准化
        if dataset.scaler_y is not None:
            valid_preds_original = dataset.scaler_y.inverse_transform(valid_preds.reshape(-1, 1)).flatten()
            valid_targets_original = dataset.scaler_y.inverse_transform(valid_targets.reshape(-1, 1)).flatten()
        else:
            valid_preds_original = valid_preds
            valid_targets_original = valid_targets
        
        mse = ((valid_preds_original - valid_targets_original) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(valid_preds_original - valid_targets_original).mean()
        
        print("\n=== Regression Results (Valid Samples Only) ===")
        print(f"Valid samples: {valid_indices.sum()} / {len(masks)}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("\n=== Regression Results ===")
        print("No valid samples for regression evaluation")


def predict(model, dataset, input_dict, device):
    """对单个样本进行预测
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象（用于标准化和反标准化）
        input_dict: 输入字典，如 {"HCl_mL": 0.5, "CH3COOH_mL": 0.5, ...}
        device: 计算设备
    
    Returns:
        dict: 包含二分类预测和回归预测的字典
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
        binary_pred, regression_pred = model(input_tensor)
        binary_pred = binary_pred.cpu().numpy()[0]
        regression_pred = regression_pred.cpu().numpy()[0, 0]
    
    # 二分类结果
    binary_cols = ["Crystallized", "FCC_Phase", "Mesoporous", "Uniform_Mesoporous", "Non_Spherical"]
    binary_results = {}
    for i, col in enumerate(binary_cols):
        binary_results[col] = {
            'probability': float(binary_pred[i]),
            'prediction': int(binary_pred[i] > 0.5)
        }
    
    # 回归结果 (反标准化)
    all_ones = all(binary_results[col]['prediction'] == 1 for col in binary_cols)
    if dataset.scaler_y is not None and all_ones:
        regression_value = dataset.scaler_y.inverse_transform([[regression_pred]])[0, 0]
    else:
        regression_value = regression_pred
    
    return {
        'binary_predictions': binary_results,
        'intensity_ratio': {
            'value': float(regression_value),
            'valid': all_ones,
            'note': 'Valid only if all binary predictions are 1' if not all_ones else 'Valid'
        }
    }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train MOF prediction model')
    parser.add_argument('--seed', type=int, default=421, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # 加载数据
    data_path = "data/subset_50.csv"
    dataset = MOFDataset(data_path)
    
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 创建模型
    model = HierarchicalMOFModel(
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
    evaluate_model(model, test_loader, dataset, device)
    
    # 保存模型
    model_save_path = "data/best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': dataset.scaler_X,
        'scaler_y': dataset.scaler_y
    }, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # 示例预测
    print("\n=== Example Prediction ===")
    example_input = {
        "HCl_mL": 0.5,
        "CH3COOH_mL": 0.5,
        "ZrCl4_mmol": 0.05,
        "HfCl4_mmol": 0.05,
        "Water_mL": 3.0
    }
    result = predict(model, dataset, example_input, device)
    print(f"Input: {example_input}")
    print(f"\nBinary Predictions:")
    for col, pred in result['binary_predictions'].items():
        print(f"  {col}: {pred['prediction']} (prob: {pred['probability']:.4f})")
    print(f"\nIntensity Ratio: {result['intensity_ratio']['value']:.4f} ({result['intensity_ratio']['note']})")


if __name__ == "__main__":
    main()
