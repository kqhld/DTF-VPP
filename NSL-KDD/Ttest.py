from scipy.stats import ttest_rel
import numpy as np

# 替换为你真实运行结果 ↓↓↓
# 假设你已经做了 5 次独立实验，以下是你模型的测试准确率和 baseline 模型的测试准确率
# your_model_acc = [0.8637, 0.8529, 0.8655, 0.8542, 0.8613]     # 你的方法的准确率（Transformer+PCA）
# baseline_acc   = [0.8339, 0.8356, 0.8372, 0.8303, 0.8319]     # Baseline（Transformer）
your_model_acc = [0.8607, 0.8529, 0.8605, 0.8542, 0.8610]     # 你的方法的F1（Transformer+PCA）
baseline_acc   = [0.8279, 0.8276, 0.8272, 0.8273, 0.8279]     # Baseline（Transformer）
# 可选：检查长度一致
assert len(your_model_acc) == len(baseline_acc), "实验次数不一致"

# 执行配对 t 检验
t_stat, p_value = ttest_rel(your_model_acc, baseline_acc)

# 打印结果
print("\n=== Paired t-test Result (Accuracy) ===")
print(f"Your model mean ± std: {np.mean(your_model_acc):.4f} ± {np.std(your_model_acc):.4f}")
print(f"Baseline mean ± std:   {np.mean(baseline_acc):.4f} ± {np.std(baseline_acc):.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

# 判断是否显著
if p_value < 0.05:
    print("✅ 显著性成立（p < 0.05）：你的方法在准确率上显著优于Baseline")
else:
    print("⚠️ 差异不显著（p >= 0.05）：你的方法提升可能是偶然波动")
