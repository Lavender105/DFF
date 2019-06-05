# Dynamic Feature Fusion for Semantic Edge Detection (DFF)
Yuan Hu, Yunpeng Chen, Xiang Li and Jiashi Feng

![overview.png](https://github.com/Lavender105/DFF/blob/master/img/overview.png)
![visualization.png](https://github.com/Lavender105/DFF/blob/master/img/visualization.png)

## Introduction
The repository contains the entire pipeline (including data preprocessing, training, testing, visualization, evaluation and demo generation, etc) for DFF using Pytorch 1.0.

We propose a novel dynamic feature fusion strategy for semantic edge detection. This is achieved by a proposed weight learner to infer proper fusion weights over multi-level features for each location of the feature map, conditioned on the specific input. We show that our model with the novel dynamic feature fusion is superior to fixed weight fusion and also the na¨ ıve location-invariant weight fusion methods, and we achieve new state-of-the-art on benchmarks Cityscapes and SBD. For more details, please refer to the [IJCAI2019](https://arxiv.org/abs/1902.09104) paper.

We also reproduce CASENet in this repository, and actually achieve higher accuracy than the original [paper](https://arxiv.org/abs/1705.09759) .
