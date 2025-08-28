import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import random
import warnings
import os

warnings.filterwarnings("ignore")

from model.Trans import TransformerModel
from model.LSTM import LSTM
from model.DNN import DNN
from model.Trans_PCA import TransformerModel1
# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 初始化模型权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
batch_size = 256
num_epochs = 200
learning_rate = 0.0005
weight_decay = 1e-4
model_name = "transformer1"  # 可选: "transformer", "dnn", "lstm", "transformer1"
seeds = [42, 123, 2023, 456, 789]  # 5次运行

# 确保模型保存目录存在
os.makedirs("model", exist_ok=True)

# ====== 数据读取和预处理 ======
df = pd.read_csv('no_Enhanced_Synthetic_Cyber_Attack_Dataset.csv')

# 时间戳处理
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp'] = df['Timestamp'].astype(np.int64) // 10**9

# IP 转换为整数
def ip_to_int(ip):
    parts = ip.split('.')
    return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])

df['Source_IP'] = df['Source_IP'].apply(ip_to_int)
df['Destination_IP'] = df['Destination_IP'].apply(ip_to_int)

# 协议编码 + 标签编码
df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])
df['Attack_Type'] = LabelEncoder().fit_transform(df['Attack_Type'])

# 特征与标签分离
label_col = 'Attack_Type'
labels = df[label_col]
features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

# 根据模型选择是否使用 PCA
if model_name == "transformer1":
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    pca = PCA(n_components=0.8)
    features = pca.fit_transform(features)
else:
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

num_columns = features.shape[1]
num_classes = len(np.unique(labels))
print(f"特征数量: {num_columns}, 类别数: {num_classes}")

# ====== 多次运行结果 ======
results = {"acc": [], "precision": [], "recall": [], "f1": []}
global_best_train_acc = 0
global_best_model_path = "model/global_best.model"

for run_id, seed in enumerate(seeds):
    print(f"\n==== Run {run_id+1}/{len(seeds)}, Seed: {seed} ====")
    set_seed(seed)

    # ===== 划分训练集(64%)、验证集(16%)、测试集(20%) =====
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
    )
    #转为tensor
    X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32).to(device),(X_train, X_val, X_test))
    y_train, y_val, y_test = map(lambda x: torch.tensor(x.values, dtype=torch.long).to(device),(y_train, y_val, y_test))

    # DataLoader
    train_loader = Data.DataLoader(Data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(Data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # 初始化模型
    if model_name == "transformer":
        model = TransformerModel(input_size=num_columns, num_classes=num_classes).to(device)
    elif model_name == "dnn":
        model = DNN(input_size=num_columns, num_classes=num_classes).to(device)
    elif model_name == "lstm":
        model = LSTM(input_size=num_columns, num_classes=num_classes).to(device)
    elif model_name == "transformer1":
        model = TransformerModel1(input_size=num_columns, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ===== 训练全部 epoch =====
    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        for data, label in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(dim=1) == label).sum().item()
        train_acc = train_correct / len(train_loader.dataset)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Run{run_id+1} Epoch{epoch+1}/{num_epochs} Train-Acc={train_acc:.4f}")

    # ===== 每个 seed 训练完成后保存该 seed 模型，并比较全局最佳 =====
    seed_model_path = f"model/run{run_id+1}_final.model"
    torch.save(model.state_dict(), seed_model_path)
    if train_acc > global_best_train_acc:
        global_best_train_acc = train_acc
        torch.save(model.state_dict(), global_best_model_path)
        print(f"[全局最佳更新] Run{run_id+1}: Train-Acc={train_acc:.4f}")

    # ===== 测试集评估 =====
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = outputs.argmax(dim=1)
        test_acc = (preds == y_test).sum().item() / y_test.size(0)
        precision = precision_score(y_test.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(y_test.cpu(), preds.cpu(), average='weighted')
        f1 = f1_score(y_test.cpu(), preds.cpu(), average='weighted')

    results["acc"].append(test_acc)
    results["precision"].append(precision)
    results["recall"].append(recall)
    results["f1"].append(f1)

    print(f"Run{run_id+1} 测试集: Acc={test_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# ===== 输出平均结果 =====
print("\n=== 5次运行平均结果 ===")
print(f"平均Acc={np.mean(results['acc']):.4f} ± {np.std(results['acc']):.4f}")
print(f"平均Precision={np.mean(results['precision']):.4f}")
print(f"平均Recall={np.mean(results['recall']):.4f}")
print(f"平均F1={np.mean(results['f1']):.4f}")

# ===== 使用全局最佳模型评估 + Macro-AUC =====
best_model = type(model)(input_size=num_columns, num_classes=num_classes).to(device)
best_model.load_state_dict(torch.load(global_best_model_path))
best_model.eval()

with torch.no_grad():
    outputs = best_model(X_test)
    preds = outputs.argmax(dim=1)
    test_acc = (preds == y_test).sum().item() / y_test.size(0)
    precision = precision_score(y_test.cpu(), preds.cpu(), average='weighted')
    recall = recall_score(y_test.cpu(), preds.cpu(), average='weighted')
    f1 = f1_score(y_test.cpu(), preds.cpu(), average='weighted')

    # Macro-AUC
    y_score = outputs.softmax(dim=1).cpu().numpy()
    y_true_bin = label_binarize(y_test.cpu().numpy(), classes=np.arange(num_classes))
    macro_auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')

print("\n=== 使用全局最佳模型 (Train-Acc最高) 的测试结果 ===")
print(f"Test Acc={test_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
      f"F1={f1:.4f}, Macro-AUC={macro_auc:.4f}")

# ===== ROC曲线绘图 =====
plt.figure(figsize=(12, 9))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_val:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Class (Global Best Model)')
plt.legend(loc='lower right', fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_auc_global_best.png', dpi=300)
plt.show()
