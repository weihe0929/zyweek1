import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体为 SimHei
rcParams['axes.unicode_minus'] = False  #显示负号
# 加载数据
file_path = 'nigerian-songs.csv'
data = pd.read_csv(file_path)  # 注意：原文件可能是CSV格式（根据扩展名调整）

# 查看数据基本信息
print(data.info())
print(data.head())

# 选择数值型特征
numerical_features = data.select_dtypes(include=['int64', 'float64'])
print("数值型特征：", numerical_features.columns.tolist())

# 检查缺失值
print("缺失值统计：\n", numerical_features.isnull().sum())

# 标准化数据
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
scaled_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)

print("数值型特征统计摘要：\n", numerical_features.describe())
# 计算相关性矩阵
corr_matrix = numerical_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("数值型特征相关性矩阵")
plt.show()


# 绘制关键特征的分布直方图
key_features = ['danceability', 'energy', 'loudness', 'tempo', 'popularity']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], kde=True)
    plt.title(f"{feature} 分布")
plt.tight_layout()
plt.show()

# 计算不同聚类数下的WSS
wss = []
cluster_range = range(1, 11)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_df)
    wss.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wss, marker='o')
plt.title("肘部法确定最佳聚类数")
plt.xlabel("聚类数")
plt.ylabel("WSS（簇内平方和）")
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

# 选择聚类数（根据肘部图）
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_df)

# 将聚类结果添加到原始数据
data['cluster'] = clusters
print("聚类分布：\n", data['cluster'].value_counts())

# 获取聚类中心
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                              columns=numerical_features.columns)
print("聚类中心点（原始特征空间）：\n", cluster_centers)

# 可视化聚类中心的关键特征
plt.figure(figsize=(12, 6))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.barplot(x=cluster_centers.index, y=cluster_centers[feature])
    plt.title(f"{feature} 聚类中心对比")
plt.tight_layout()
plt.show()


# 使用PCA降维到2D以便可视化
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

# 将PCA结果添加到数据中
data['pca1'] = pca_result[:, 0]
data['pca2'] = pca_result[:, 1]

# 绘制聚类散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis', alpha=0.7)
plt.title(f"K-Means聚类结果（{optimal_clusters}个簇，PCA降维）")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# 计算每个聚类的特征均值
cluster_means = data.groupby('cluster')[numerical_features.columns].mean()
print("每个聚类的特征均值：\n", cluster_means)

# 可视化聚类特征均值
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap='YlGnBu', annot=True, fmt=".2f")
plt.title("每个聚类的特征均值热力图")
plt.xlabel("聚类")
plt.ylabel("特征")
plt.show()