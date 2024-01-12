# MusicGenreClassification 基于机器学习与深度学习的音乐风格分类与预测——特征学习与模型性能

本项目为《机器学习》课程项目，使用`FMA_small`数据集训练传统机器学习模型与深度学习模型，进行音乐风格的分类与预测。项目结构如下：

## data_correction

由于数据集内部分音乐片段损坏，导致音乐片段无法正常读取和提取特征。该目录下的音乐片段用于替换同名的已损坏音乐片段。

## dataset_process

该目录是用于深度学习训练的Jupyter Notebook的预处理部分，从`FMA: A Dataset For Music Analysis`项目中克隆而来并经过修改，用于对比性能等。

## neural_network

该目录是基于`CNN`和`CRNN`的深度学习模型代码文件部分，包含运行结果、混淆矩阵等。

## traditional_method

该目录是基于隐马尔可夫、逻辑回归和支持向量机的传统机器学习模型代码文件部分，包含运行结果、混淆矩阵、超参数组合和ROC曲线等。
