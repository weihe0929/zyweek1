# zyweek1

## 项目概述
本实验使用无监督学习技术对尼日利亚歌曲的音频特征进行聚类分析，旨在发现音乐数据中的自然分组模式，为音乐推荐系统和市场细分提供数据支持。
## 数据集
### 数据文件：nigerian-songs.csv   

- name: 歌曲的名称。
album: 歌曲所属的专辑名称。
artist: 演唱该歌曲的艺术家。
artist_top_genre: 艺术家的主要音乐流派。
release_date: 歌曲的发布日期。
length: 歌曲的长度，以毫秒为单位。
popularity: 歌曲的流行度，可能是一个从0到100的评分。
danceability: 歌曲的舞蹈性，一个衡量歌曲适合跳舞程度的指标，值越高表示越适合跳舞。
acousticness: 歌曲的声学特性，值越高表示越具有声学特性。
energy: 歌曲的能量水平，值越高表示歌曲越有活力。
instrumentalness: 歌曲的器乐性，值越高表示歌曲越可能是器乐曲。
liveness: 歌曲的现场感，值越高表示歌曲可能是在现场录制的。
loudness: 歌曲的响度，以分贝为单位，值越低表示歌曲越安静。
speechiness: 歌曲的说话性，值越高表示歌曲中说话的成分越多。
tempo: 歌曲的节奏，以每分钟节拍数(BPM)表示。
time_signature: 歌曲的拍号，表示每小节的拍数。
## 实验思路
数据理解与预处理
-数据加载
读取nigerian-songs.csv文件，检查字段类型、缺失值（如popularity是否有空值）。

-特征选择
仅保留数值型特征（如danceability、energy、loudness等），剔除文本或分类变量（如歌曲名、艺术家）。

-数据标准化
使用StandardScaler对特征标准化，消除量纲影响（如tempo和loudness的单位差异）。

探索性分析
-相关性分析
通过热图观察特征间的相关性（如energy和loudness通常正相关）。

-分布可视化
绘制直方图检查特征分布（如popularity是否偏态，是否需要对数变换）。

聚类模型构建
-确定最佳聚类数
使用肘部法（Elbow Method）计算不同K值对应的WSS（簇内平方和），选择拐点作为K值（如K=5）。

-模型训练
用K-Means算法拟合标准化后的数据，生成聚类标签。

结果验证与分析
-聚类中心解读
逆标准化聚类中心，分析每个簇的典型特征（例如：簇0可能是“高能量快节奏歌曲”）。

-降维可视化
通过PCA将数据降至2维，绘制散点图观察聚类分离效果。

-特征对比
用热图对比不同簇的特征均值，明确各簇的区分度。

## 实验结果
![image](https://github.com/user-attachments/assets/6ad57d46-acc6-4b18-b011-9c2c3d369303)
