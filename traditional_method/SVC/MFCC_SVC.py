from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

mfcc_data_path = "D:\\dataset\\mfcc_re.csv"
mfcc_data = pd.read_csv(mfcc_data_path)
print(mfcc_data)

# 分离特征和标签
MFCC_X = mfcc_data.drop(['track_id', 'genre'], axis=1)
MFCC_y = mfcc_data['genre']

# 划分数据集为训练集和测试集
MFCC_X_train, MFCC_X_test, MFCC_y_train, MFCC_y_test = train_test_split(MFCC_X, MFCC_y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
MFCC_X_train_scaled = scaler.fit_transform(MFCC_X_train)
MFCC_X_test_scaled = scaler.transform(MFCC_X_test)

# 训练逻辑回归模型
MFCC_best_params = {'C': '3.0', 'degree': '1', 'gamma': '0.01', 'kernel': 'rbf'}
MFCC_model = SVC(C=3, degree=1, gamma=0.01, kernel='rbf', random_state=42, probability=True)
MFCC_model.fit(MFCC_X_train_scaled, MFCC_y_train)

# 5. 预测和评估
MFCC_y_pred = MFCC_model.predict(MFCC_X_test_scaled)
MFCC_accuracy = accuracy_score(MFCC_y_test, MFCC_y_pred)
print(classification_report(MFCC_y_test, MFCC_y_pred))
conf_matrix = confusion_matrix(MFCC_y_test, MFCC_y_pred, normalize='true')
print(conf_matrix)

MFCC_output_result_path = "MFCC_SVC_Result.txt"
with open(MFCC_output_result_path, "w") as output_file:
    output_file.write("Classification Report\n")
    output_file.write(str(classification_report(MFCC_y_test, MFCC_y_pred)))
    output_file.write("\n")
    output_file.write("Confusion Matrix\n")
    output_file.write(str(conf_matrix))
    output_file.write("\n")
    output_file.write("Params\n")
    output_file.write(str(MFCC_best_params))

# 计算每个类别的ROC曲线和AUC分数
fpr = dict()
tpr = dict()
roc_auc = dict()

label_mapping = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
                 'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}

y_test_numeric = np.array([label_mapping[label] for label in MFCC_y_test])
y_test_binary = label_binarize(y_test_numeric, classes=list(label_mapping.values()))

y_pred_proba = MFCC_model.predict_proba(MFCC_X_test)

print(y_test_binary)
print(y_pred_proba)

for i in range(len(label_mapping)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
plt.figure(figsize=(12, 12), dpi=100)
for label in label_mapping:
    plt.plot(fpr[label_mapping[label]], tpr[label_mapping[label]],
             label=f'{label} (AUC = {roc_auc[label_mapping[label]]:.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Multiclass ROC Curve')
plt.legend()
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(12, 12), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_mapping.keys())
disp.plot(cmap='Blues', values_format=".2f", xticks_rotation=30)
plt.title('Normalized Confusion Matrix')
plt.show()
