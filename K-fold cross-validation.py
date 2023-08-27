'''
author:zhicong
time:2021/12/20 12:01
'''
import pandas as pd
import numpy as np
import tensorflow as tf
print('Tensorflow vision is:{}'.format(tf.__version__))

if __name__ == '__main__':
    # 留出法划分数据集
    # 优点:思路简单清晰
    # 缺点：1、模型的泛化能力很大程度取决于划分方法 2、无法充分利用数据进而导致模型欠拟合，即测试数据没有用于训练模型
    data = pd.read_csv('./dataset.csv')  # 读取dataset.csv中的数据
    # 数据集dataset.csv是一个1000行，3列的数据，第一列为样本ID，第二列为样本值value，第三列为样本标签label
    # 当样本值大于0时label=1，当样本值小于0时label=0

    ID = np.array(data.ID)  # 读取样本ID
    value = np.array(data.value)  # 读取样本value
    label = np.array(data.label)  # 读取样本标签 正数-->1;负数-->0

    dataset = tf.data.Dataset.from_tensor_slices((ID, label))  # 调用tensorflow下from_tensor_slices方法创建数据集
    dataset = dataset.shuffle(1000)  # 对数据集进行乱序
    length = len(ID)
    train_length = int(0.3*length)
    test_length = length-train_length
    train_dataset = dataset.skip(train_length)  # 取数据集的前70%为训练数据集
    test_dataset = dataset.take(train_length)  # 取数据集的后30%为测试数据集

    train_list = list(train_dataset.as_numpy_iterator())
    test_list = list(test_dataset.as_numpy_iterator())

    count = 0
    for i, j in enumerate(train_list):
        if j[1] == 1:
            count = count+1
    train_1 = count
    train_1_proportion = train_1/len(train_list)

    count = 0
    for i, j in enumerate(train_list):
        if j[1] == 0:
            count = count+1
    train_0 = count
    train_0_proportion = train_0/len(train_list)

    count = 0
    for i, j in enumerate(test_list):
        if j[1] == 1:
            count = count+1
    test_1 = count
    test_1_proportion = test_1/len(test_list)

    count = 0
    for i, j in enumerate(test_list):
        if j[1] == 0:
            count = count+1
    test_0 = count
    test_0_proportion = test_0/len(test_list)



    # K折交叉验证法划分数据集
    # 将数据集等分成K份，，轮流将其中1份用于验证(测试)模型,其余K-1份用于训练模型
    def k_cross_validation(dataset, k):

        length = len(dataset)
        batch_size = int(length / k)
        dataset = dataset.batch(batch_size)  # 调用tensorflow框架下batch方法，将数据集划分成一层一层的batch
        list_ = list(dataset.as_numpy_iterator())  # 将数据集中的数据写进列表中取
        for i in range(k):
            count = 0
            dataset = tf.data.Dataset.from_tensor_slices((list_[i][0], list_[i][1]))
            for j in range(len(list_[i][1])):
                if list_[i][1][j] == 1:
                    count = count + 1
            print('第{}个数据集为\n{}'.format(i + 1, list(dataset.as_numpy_iterator())))
            print('该数据集中标签为1的比例为{}标签为0的比例为{}'.format(count/len(list_[i][1]), 1-count/len(list_[i][1])))
            # print('-----------------')


    k_cross_validation(dataset, 10)  # k=10 将数据集10等分

