import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from model.Trans_PCA import TransformerModel1
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =================== 基础配置 ====================
model_path = '../model/best.model'
model_type = "Transformer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 数据预处理 ====================
df_train = pd.read_csv('NSL_KDD_Train22.csv')
df_test = pd.read_csv('NSL_KDD_Test22.csv')
df_train.columns = list(range(43))
df_test.columns = list(range(43))
train_labels = set(df_train[42])
df_test = df_test[df_test[42].isin(train_labels)]
df = pd.concat([df_train, df_test])
features = df.drop(columns=[42])
labels = df[42]
le = LabelEncoder()
labels = le.fit_transform(labels).astype(np.int64)
for col in [1, 2, 3]:
    features[col] = le.fit_transform(features[col])
scaler = StandardScaler()
features = scaler.fit_transform(features)
pca = PCA(n_components=25)
features_pca = pca.fit_transform(features)
input_size = features_pca.shape[1]
num_classes = len(np.unique(labels))

# 转为 tensor
features_pca = torch.tensor(features_pca, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.long).to(device)
x_train_full, x_test = torch.split(features_pca, [len(df_train), len(df_test)])
y_train_full, y_test = torch.split(labels, [len(df_train), len(df_test)])

# =================== 模型构建 ====================
if model_type == "Transformer":
    model = TransformerModel1(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# =================== FGSM 攻击定义 ====================
def fgsm_attack(model, loss_fn, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data.detach()

# =================== 多ε攻击实验 + 绘图 ====================
epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
acc_list = []
f1_list = []

loss_fn = torch.nn.CrossEntropyLoss()

def evaluate(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        preds = outputs.argmax(dim=1)
    acc = (preds == y).sum().item() / y.size(0)
    f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
    return acc, f1

for eps in epsilons:
    if eps == 0.0:
        x_eval = x_test.clone().detach()
    else:
        x_eval = fgsm_attack(model, loss_fn, x_test.clone().detach(), y_test.clone().detach(), eps)
    acc, f1 = evaluate(model, x_eval, y_test)
    acc_list.append(acc)
    f1_list.append(f1)
    print(f"ε={eps:.2f} | Accuracy={acc:.4f}, F1={f1:.4f}")

# =================== 绘图 ====================
plt.figure(figsize=(8, 6))
plt.plot(epsilons, acc_list, marker='o', label='Accuracy')
plt.plot(epsilons, f1_list, marker='s', label='F1 Score')
plt.xlabel("FGSM Epsilon (ε)")
plt.ylabel("Score")
plt.title("Model Performance under FGSM Attack")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
