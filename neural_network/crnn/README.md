原作者：https://github.com/priya-dwivedi/Music_Genre_Classification
这个仓库包含了音乐流派识别项目的代码。

依赖包

Python 3.6.5
Tensorflow - 1.7.0
Keras - 2.2.4
Numpy、Pandas、Matplotlib
Librosa - 0.6.2
原始数据

下载FMA Small数据集：https://github.com/mdeff/fma。原始数据为8GB，包含来自8000首歌曲的音频 + 元数据，包括MFCC等特征。

预处理数据

原始音频已转换为梅尔频谱图并保存为pickle格式。有三个文件用于训练、验证和测试.

在运行任何带有神经网络的notebook之前，请确保下载了这些文件。

代码Notebooks

Explore data, convert raw audio into spectrograms and pickle them

load_fma_dataset: 加载fma_dataset并对其进行探索。
Plot_Spectograms: 绘制8种不同流派的频谱图。
convert_to_npz: 加载原始音频，将每个文件转换为频谱图，并将结果保存为pickle格式，以便于训练模型。这些数据集的输出在上述的Google Drive链接中。
Building models

baseline_model_fma: 此模型使用tracks.csv中的元数据加载MFCC特征，并构建一个SVC分类器。
CRNN_model: 此notebook使用压缩的频谱图构建了一个Keras中的CRNN模型。
模型文件夹中包含了这个模型的训练权重。

This repository contains the code for Music Genre Recognition project

## Packages
* Python 3.6.5
* Tensorflow - 1.7.0
* Keras - 2.2.4
* Numpy, Pandas, Matplotlib
* Librosa - 0.6.2