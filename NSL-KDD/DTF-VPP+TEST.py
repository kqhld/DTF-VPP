import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import random
import warnings
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
num_runs = 5
seeds = [42, 123, 2023, 456, 789]

# ===== 数据加载 =====
df_train = pd.read_csv('NSL_KDD_Train22.csv')
df_test = pd.read_csv('NSL_KDD_Test22.csv')
df_train.columns = list(range(43))
df_test.columns = list(range(43))
train_rows = len(df_train)

# 删除测试集中未在训练集中出现的标签
train_labels = set(df_train[42])
df_test = df_test[df_test[42].isin(train_labels)]

# 合并数据
df = pd.concat([df_train, df_test])
features = df.drop(columns=[42])
labels = df[42]

# 编码
le = LabelEncoder()
labels = le.fit_transform(labels).astype(np.int64)
for col in [1, 2, 3]:
    features[col] = le.fit_transform(features[col])

# 根据模型选择是否使用 PCA
if model_name == "transformer1":
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    pca = PCA(n_components=25)
    features = pca.fit_transform(features)
else:
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

num_columns = features.shape[1]
num_classes = len(np.unique(labels))
print(f"特征数量: {num_columns}, 类别数: {num_classes}")

# 转为 tensor
features = torch.tensor(features, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.long).to(device)

# 划分训练/测试集
x_train_full, x_test = torch.split(features, [train_rows, len(features) - train_rows])
y_train_full, y_test = torch.split(labels, [train_rows, len(features) - train_rows])

# 存储指标
test_acc_list, test_precision_list, test_recall_list, test_f1_list = [], [], [], []

best_model_path = "../model/bestmodel_nslkdd"  # 固定文件名
best_train_acc = 0
for run_id, seed in enumerate(seeds):
    print(f"\n==== Run {run_id+1}/{num_runs}, Seed: {seed} ====")
    set_seed(seed)

    # 训练验证划分（仅用于 DataLoader，不用 val_acc 保存）
    dataset_full = Data.TensorDataset(x_train_full, y_train_full)
    train_loader = Data.DataLoader(dataset_full, batch_size=batch_size, shuffle=True)

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

    # ===== 训练 =====
    for epoch in range(num_epochs):
        model.train()
        train_correct, train_loss = 0, 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == label).sum().item()
        train_acc = train_correct / len(train_loader.dataset)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[保存] 当前最佳 Train-Acc={best_train_acc:.4f}，已保存 best.model")
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}")

# ===== 测试集评估 =====
model.load_state_dict(torch.load(best_model_path))  # 使用训练集最优模型
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    preds = outputs.argmax(dim=1)
    test_acc = (preds == y_test).sum().item() / y_test.size(0)
    precision = precision_score(y_test.cpu(), preds.cpu(), average='weighted')
    recall = recall_score(y_test.cpu(), preds.cpu(), average='weighted')
    f1 = f1_score(y_test.cpu(), preds.cpu(), average='weighted')

print("\n=== 使用 best.model (Train-Acc 最优) 在测试集上评估 ===")
print(f"Test Acc={test_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# ===== ROC-AUC 绘图 =====
best_model = None
if model_name == "transformer":
    best_model = TransformerModel(input_size=num_columns, num_classes=num_classes).to(device)
elif model_name == "dnn":
    best_model = DNN(input_size=num_columns, num_classes=num_classes).to(device)
elif model_name == "lstm":
    best_model = LSTM(input_size=num_columns, num_classes=num_classes).to(device)
elif model_name == "transformer1":
    best_model = TransformerModel1(input_size=num_columns, num_classes=num_classes).to(device)

best_model.load_state_dict(torch.load(best_model_path))  # 用训练集最优模型
best_model.eval()

with torch.no_grad():
    y_score = best_model(x_test).softmax(dim=1).cpu().numpy()
    y_true_bin = label_binarize(y_test.cpu().numpy(), classes=np.arange(num_classes))

plt.figure(figsize=(12, 9))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Class (using best.model)')
plt.legend(loc='lower right', fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_auc_best_model.png', dpi=300)
plt.show()
