'''
author:zhicong
time:2022/1/2 21:43
'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, mutual_info_score, homogeneity_score, completeness_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, SpectralBiclustering, FeatureAgglomeration

# 聚类结果可视化
def visualization(x, pred, name_):
    centers = [[1, 1], [-1, -1], [1, -1]]
    cmap = matplotlib.colors.ListedColormap(['b', 'g', 'r'])
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    plt.cla()
    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(x[labels == label, 0].mean(),
                  x[labels == label, 1].mean() + 1.5,
                  x[labels == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    y = np.choose(pred, [1, 2, 0]).astype(np.float)
    ax.scatter(x[:, 0], x[:, 2], x[:, 1], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.suptitle('{}\n{}'.format(name_, 'result'))
    plt.show()


if __name__ == '__main__':
    # 获取sklearn内置数据集
    dataset = load_iris()
    labels = dataset.target
    x = dataset.data
    # 对特征进行PCA降维
    pca = PCA(n_components=3)
    pca.fit(x)

    # 创建模型
    model_kmeans = KMeans(n_clusters=3)  # 创建具有3聚类中心的Kmeans模型
    model_dbscan = DBSCAN(eps=1, min_samples=4)  # 创建一个邻域最大距离为1.5,核心点样本数为4的DBSCAN模型
    model_agglomerative = AgglomerativeClustering(n_clusters=3, affinity='euclidean')  # 凝聚聚类,聚类数目为3,相似度计算为欧式距离

    # 聚类实现获取聚类标签
    pred_kmeans = model_kmeans.fit_predict(x)
    pred_dbscan = model_dbscan.fit_predict(x)
    pred_agglomerative = model_agglomerative.fit_predict(x)


    # Kmeans模型评价指标
    kmeans_mut_info = mutual_info_score(labels, pred_kmeans)  # 互信息
    kmeans_sil_score = silhouette_score(x, pred_kmeans)  # 轮廓系数
    kmeans_homo_score = homogeneity_score(labels, pred_kmeans)  # 聚类标签的同质性度量
    kmeans_com_score = completeness_score(labels, pred_kmeans)  # 聚类标签的完整性度量
    kmeans_rand_score = adjusted_rand_score(labels, pred_kmeans)  # 随机兰德调整指数

    # dbscan密度聚类模型评价指标
    dbscan_mut_info = mutual_info_score(labels, pred_dbscan)
    dbscan_sil_score = silhouette_score(x, pred_dbscan)
    dbscan_homo_score = homogeneity_score(labels, pred_dbscan)
    dbscan_com_score = completeness_score(labels, pred_dbscan)
    dbscan_rand_score = adjusted_rand_score(labels, pred_dbscan)

    # agglomeration凝聚聚类模型评价指标
    agglomerative_mut_info = mutual_info_score(labels, pred_agglomerative)
    agglomerative_sil_score = silhouette_score(x, pred_agglomerative)
    agglomerative_homo_score = homogeneity_score(labels, pred_agglomerative)
    agglomerative_com_score = completeness_score(labels, pred_agglomerative)
    agglomerative_rand_score = adjusted_rand_score(labels, pred_agglomerative)

    # 打印输出
    print('Kmeans聚类模型的评价指标如下:\n互信息:{:.3f},轮廓系数:{:.3f},\n聚类标签的同质性度量:{:.3f},聚类标签的完整性度量:{:.3f},随机兰德调整指数:{:.3f}'
          .format(kmeans_mut_info, kmeans_sil_score, kmeans_homo_score, kmeans_com_score, kmeans_rand_score))

    print('dbscan密度聚类模型的评价指标如下:\n互信息:{:.3f},轮廓系数:{:.3f},\n聚类标签的同质性度量:{:.3f},聚类标签的完整性度量:{:.3f},随机兰德调整指数:{:.3f}'
          .format(dbscan_mut_info, dbscan_sil_score, dbscan_homo_score, dbscan_com_score, dbscan_rand_score))

    print('凝聚聚类模型的评价指标如下:\n互信息:{:.3f},轮廓系数:{:.3f},\n聚类标签的同质性度量:{:.3f},聚类标签的完整性度量:{:.3f},随机兰德调整指数:{:.3f}'
          .format(agglomerative_mut_info, agglomerative_sil_score, agglomerative_homo_score, agglomerative_com_score, agglomerative_rand_score))

    visualization(x, pred_kmeans, 'Kmeans聚类')
    visualization(x, pred_dbscan, 'dbscan聚类')
    visualization(x, pred_agglomerative, '凝聚聚类')
