'''
author:zhicong
time:2021/12/23 16:49
'''
import tensorflow as tf
import numpy as np
import pandas
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PCA(data, k):
    # PCA降维的步骤
    # 1、计算样本协方差矩阵S 2、对S进行特征值分解得到特征值λ1,λ2……λn,取最大λ11>λ21>λ31>……>λd1
    # 3、λ11>λ21>λ31>……>λd1特征值所对应的特征向量
    mean_0 = np.mean(data, axis=0)  # 求每一列的均值
    data = data - mean_0  # 去中心化，使每一列的特征均值都为0

    S = np.cov(data, rowvar=False)  # 计算去中心化后数据的协方差矩阵,
    lambda_, lambda_vector = np.linalg.eig(S)  # 对协方差矩阵进行特征值分解,得到特征值和特征向量

    index = np.argsort(lambda_)  # 返回特征值从小到大元素的索引 index=[3 2 1 0]
    index = index[-1:-k-1:-1]  # 取三个最大特征值所对应的索引
    lambda_vector = lambda_vector[:, index]  # 得到一组正交基

    data_k_dim = np.dot(data, lambda_vector)  # 降维后的数据(三维)

    data_k_dim = data_k_dim.T
    return data_k_dim

if __name__ == '__main__':
    data = pandas.read_csv('./Iris.csv')  # 读取数据集
    column_1 = np.array(data.SepalLengthCm)
    column_2 = np.array(data.SepalWidthCm)
    column_3 = np.array(data.PetalLengthCm)
    column_4 = np.array(data.PetalWidthCm)
    column_5 = list(data.Species)
    # print(column_5)
    for i in range(len(column_5)):
        if column_5[i] == 'Iris-setosa':
            column_5[i] = 0
        if column_5[i] == 'Iris-versicolor':
            column_5[i] = 1
        if column_5[i] == 'Iris-virginica':
            column_5[i] = 2
    column_5 = np.array(column_5)
    labels = column_5
    data = np.array([column_1, column_2, column_3, column_4])  # 得到一个形状为(4, 150)的二维数组，每一行对应原数据集中的每一列特征
    data = np.transpose(data)  # 转置后得到与原数据集完全相同的二维ndarray
    data_3_dim = PCA(data, k=3)

    # 将经过PCA降维后三维数据可视化
    centers = [[1, 1], [-1, -1], [1, -1]]
    fig = plt.figure(1,figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    plt.cla()

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(data_3_dim.T[labels == label, 0].mean(),
                  data_3_dim.T[labels == label, 1].mean() + 1.5,
                  data_3_dim.T[labels == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    y = np.choose(labels, [1, 2, 0]).astype(np.float)
    ax.scatter(data_3_dim.T[:, 0], data_3_dim.T[:, 1], data_3_dim.T[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.suptitle('result')
    plt.show()

    # 使用降维后的数据划分数据集,70%的训练数据,30%的测试数据
    dataset = tf.data.Dataset.from_tensor_slices((data_3_dim.T, labels)).shuffle(150)  # 封装数据集,同时对数据进行乱序
    train_dataset = dataset.take(int(0.7*len(data)))
    test_dataset = dataset.skip(int(0.7*len(data)))

    train_list = list(train_dataset.as_numpy_iterator())
    test_list = list(test_dataset.as_numpy_iterator())
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(int(0.7*(len(dataset)))):
        train_data.append(train_list[i][0])
        train_labels.append(train_list[i][1])
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    for i in range(int(0.3*(len(dataset)))):
        test_data.append(test_list[i][0])
        test_labels.append(test_list[i][1])
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # 使用原始数据划分数据集
    x = data
    y = labels
    raw_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(150)
    raw_train_dataset = raw_dataset.take(int(0.7*len(raw_dataset)))
    raw_test_dataset = raw_dataset.skip(int(0.7*len(raw_dataset)))

    raw_train_list = list(raw_train_dataset.as_numpy_iterator())
    raw_test_list = list(raw_test_dataset.as_numpy_iterator())
    raw_train_data = []
    raw_train_labels = []
    raw_test_data = []
    raw_test_labels = []
    for i in range(int(0.7*(len(raw_dataset)))):
        raw_train_data.append(raw_train_list[i][0])
        raw_train_labels.append(raw_train_list[i][1])
    raw_train_data = np.array(raw_train_data)
    raw_train_labels = np.array(raw_train_labels)

    for i in range(int(0.3*(len(raw_dataset)))):
        raw_test_data.append(raw_test_list[i][0])
        raw_test_labels.append(raw_test_list[i][1])
    raw_test_data = np.array(raw_test_data)
    raw_test_labels = np.array(raw_test_labels)

    # 模型搭建
    model_PCA = svm.SVC(C=1,                        # 误差惩罚系数
                    kernel='sigmoid',                   # 核函数
                    decision_function_shape='ovr',  # 决策函数
                    gamma='scale')

    model_raw = svm.SVC(C=1,                        # 误差惩罚系数
                    kernel='rbf',                  # 核函数
                    decision_function_shape='ovr',  # 决策函数
                    gamma='scale')
    # 模型训练
    model_PCA.fit(train_data, train_labels)  # 使用降维训练数据对模型进行训练
    model_raw.fit(raw_train_data, raw_train_labels)  # 使用原始训练数据对模型进行训练

    # 预测
    pred = model_PCA.predict(test_data)  # 使用训练好的模型对降维测试数据进行预测
    raw_pred = model_raw.predict(raw_test_data)  # 使用训练好的模型对原始测试数据进行预测

    # 评价指标
    precision = precision_score(test_labels, pred, average='macro')  # 精度
    recall = recall_score(test_labels, pred, average='macro')  # 召回率
    accuracy_score_ = accuracy_score(test_labels, pred)  # 精度分类得分
    F_1 = f1_score(test_labels, pred, average='macro')  # F1分数
    confusion = confusion_matrix(test_labels, pred)


    raw_precision = precision_score(raw_test_labels, raw_pred, average='macro')  # 精度
    raw_recall = recall_score(raw_test_labels, raw_pred, average='macro')  # 召回率
    raw_accuracy_score = accuracy_score(raw_test_labels, raw_pred, normalize=False)  # 精度分类得分
    raw_F_1 = f1_score(raw_test_labels, raw_pred, average='micro')  # F1分数
    raw_confusion = confusion_matrix(raw_test_labels, raw_pred)

    # 打印输出
    print('使用原始数据训练模型得到的预测标签:\n{}'.format(raw_pred))
    print('原标签\n{}'.format(raw_test_labels))
    print()
    print('使用PCA降维后的数据训练模型得到的预测标签:\n{}'.format(pred))
    print('原标签\n{}'.format(test_labels))
    print()
    print('原始数据训练SVM模型度量指标:\n精度为:{:.3f},召回率为:{:.3f},精度分类得分为:{:.3f},F1分数为:{:.3f}'.format(raw_precision, raw_recall, raw_accuracy_score, raw_F_1))
    print()
    print('PCA降维后数据训练SVM模型度量指标:\n精度为:{:.3f},召回率为:{:.3f},精度分类得分为:{:.3f}2,F1分数为:{:.3f}'.format(precision, recall, accuracy_score_, F_1))
    print()
    print('原数据训练模型得到的混淆矩阵\n{}'.format(raw_confusion))
    print()
    print('经PCA降维后数据训练模型得到的混淆矩阵\n{}'.format(confusion))
    print()











