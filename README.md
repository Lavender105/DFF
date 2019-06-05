# Dynamic Feature Fusion for Semantic Edge Detection (DFF)
Yuan Hu, Yunpeng Chen, Xiang Li and Jiashi Feng

![overview.png](https://github.com/Lavender105/DFF/blob/master/img/overview.png)
![visualization.png](https://github.com/Lavender105/DFF/blob/master/img/visualization.png)

## Introduction
The repository contains the entire pipeline (including data preprocessing, training, testing, visualization, evaluation and demo generation, etc) for DFF using Pytorch 1.0.

We propose a novel dynamic feature fusion strategy for semantic edge detection. This is achieved by a proposed weight learner to infer proper fusion weights over multi-level features for each location of the feature map, conditioned on the specific input. We show that our model with the novel dynamic feature fusion is superior to fixed weight fusion and also the na¨ ıve location-invariant weight fusion methods, and we achieve new state-of-the-art on benchmarks Cityscapes and SBD. For more details, please refer to the [IJCAI2019](https://arxiv.org/abs/1902.09104) paper.

We also reproduce CASENet in this repository, and actually achieve higher accuracy than the original [paper](https://arxiv.org/abs/1705.09759) .

## Installation
Check INSTALL.md（超链接） for installation instructions.

## Usage
### 1. Preprocessing
Cityscapes Data: In this part, we assume you are in the directory $DFF_ROOT/data/cityscapes-preprocess/. Note that in this repository, all Cityscapes pipelines are instance-sensitive only.

(1) Download the files gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip and leftImg8bit_demoVideo.zip from the Cityscapes website（超链接） to data_orig/, and unzip them:
```
unzip data_orig/gtFine_trainvaltest.zip -d data_orig && rm data_orig/gtFine_trainvaltest.zip
unzip data_orig/leftImg8bit_trainvaltest.zip -d data_orig && rm data_orig/leftImg8bit_trainvaltest.zip
unzip data_orig/leftImg8bit_demoVideo.zip -d data_orig && rm data_orig/leftImg8bit_demoVideo.zip
```

(2) Generate .png training edge labels by running the following command:
```
# In Matlab Command Window
run code/demoPreproc_gen_png_label.m
```

This will create instance-sensitive edge labels for network training in data_proc/.

(3) Generate edge ground truths for evaluation by running the following command:
```
# In Matlab Command Window
run code/demoGenGT.m
```

This will create two folders (gt_thin/ and gt_raw/) in the directory of gt_eval/, containing the thinned and unthinned evaluation ground truths.

SBD Data: In this part, we assume you are in the directory $DFF_ROOT/data/sbd-preprocess/.

(1) Download the SBD dataset from Google Drive | Baidu Yun, and place the tarball sbd.tar.gz in data_orig/. Run the following command:
tar -xvzf data_orig/sbd.tar.gz -C data_orig && rm data_orig/sbd.tar.gz

(2) Perform data augmentation and generate .png training edge labels by running the following command:
```
# In Matlab Command Window
run code/demoPreproc.m
```

This will create augmented images and their instance-sensitive(inst)/non-instance-sensitive(cls) edge labels for network training in data_proc/.

(3) Generate edge ground truths for evaluation by running the following command:
```
# In Matlab Command Window
run code/demoGenGT.m
```

This will create two folders (gt_orig_thin/ and gt_orig_raw/) in the directory of gt_eval/, containing the thinned and unthinned evaluation ground truths from the original SBD data.

.png edge label explanation
we generate .png multi-label ground truth for Cityscapes and SBD dataset. A png image has three channels (R, G, B) and each channel has 8 bits. We use a single bit to encode a category. For Cityscapes dataset, the last 19 bits are used to encode 19 categories. Specifically, 8 bits for B channel, 8 bits for G channel, and 3 bits for R channel. For SBD dataset, the last 20 bits are used to encode 20 categories. Specifically, 8 bits for B channel, 8 bits for G channel and 4 bits for R channel.

For example, for cityscapes dataset, a pixel associated with car and building will be encoded as follows:
```
00000001 00000000 00100000
       R              G              B
```
where 0 indicates non-edge, and 1 indicates edge.

2.Training
In this part, we assume you are in the directory $DFF_ROOT/exps/.
Train DFF on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1 ,2,3
python train.py --dataset cityscapes --model dff --backbone resnet50 --checkname dff  --base-size 640 --crop-size 640 --epochs 200 --batch-size 8 --lr 0.08 --workers 8

Train CASENet on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --dataset cityscapes --model casenet --backbone resnet50 --checkname dff  --base-size 640 --crop-size 640 --epochs 200 --batch-size 8 --lr 0.08 --workers 8

Train DFF on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --dataset sbd --model dff --backbone resnet50 --checkname dff  --base-size 352 --crop-size 352 --epochs 10 --batch-size 16 --lr 0.05 --workers 8

Train DFF on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --dataset sbd --model casenet --backbone resnet50 --checkname dff  --base-size 352 --crop-size 352 --epochs 10 --batch-size 16 --lr 0.05 --workers 8

3. Testing
In this part, we assume you are in the directory $DFF_ROOT/exps/.
Test DFF on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset cityscapes --model dff --checkname dff  --resume-dir runs/cityscapes/dff/dff --workers 8 --backbone resnet50--eval

Test CASENet on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset cityscapes --model casenet --checkname casenet  --resume-dir runs/cityscapes/casenet/casenet --workers 8 --backbone resnet50 --eval

Test DFF on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset SBD --model dff --checkname dff  --resume-dir runs/sbd/dff/dff --workers 8 --backbone resnet50 --eval

Test DFF on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset SBD --model casenet --checkname casenet  --resume-dir runs/sbd/casenet/casenet --workers 8 --backbone resnet50 --eval

4. Visualization
In this part, we assume you are in the directory $DFF_ROOT/exps/.
Visualize DFF on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset cityscapes --model dff --checkname dff  --resume-dir runs/cityscapes/dff/dff --workers 8 --backbone resnet50

Visualize CASENet on Cityscapes:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset cityscapes --model casenet --checkname casenet  --resume-dir runs/cityscapes/casenet/casenet --workers 8 --backbone resnet50

Visualize DFF on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset SBD --model dff --checkname dff  --resume-dir runs/sbd/dff/dff --workers 8 --backbone resnet50

Visualize CASENet on SBD:
conda activate dff-master
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test.py --dataset SBD --model casenet --checkname casenet  --resume-dir runs/sbd/casenet/casenet --workers 8 --backbone resnet50

5. Evaluation
In this part, we assume you are in the directory $DFF_ROOT/lib/matlab/eval.

(1) To perform batch evaluation of results on SBD and Cityscapes, run the following command:
# In Matlab Command Window
run demoBatchEval.m

This will generate and store evaluation results in the corresponding directories. You may also choose to evaluate certain portion of the results, by commenting the other portions of the code.

(2) To plot the PR curves of the results on SBD and Cityscapes, run the following command upon finishing the evaluation:
# In Matlab Command Window
run demoGenPR.m

This will take the stored evaluation results as input, summarize the MF/AP scores of comparing methods, and generate class-wise precision-recall curves.

6. Demo Generation
In this part, we assume you are in the directory $DFF_ROOT/lib/matlab/utils.

(1) To perform batch evaluation of results on SBD and Cityscapes, run the following command:
# In Matlab Command Window
run demoVisualizeGT.m

This will generate colored visualizations of the SBD and Cityscapes ground truths.

(2) To generate demo videos on Cityscapes, run the following command upon finishing the visualization:
# In Matlab Command Window
run demoMakeVideo.m

This will generate video files of DFF predictions and comparison with reproduced CASENet on Cityscapes video sequences.

Video Demo
We have released a demo video of DFF on Youtube and Bilibili. Click the image below to and watch the video.

Note
The Matlab code is modified from SEAL（超链接）, and the Pytorch code is modified from PyTorch-Encoding（超链接）.

Citation
If DFF is useful for your research, please consider citing:
@article{hu2019dynamic,
  title={Dynamic Feature Fusion for Semantic Edge Detection},
  author={Hu, Yuan and Chen, Yunpeng and Li, Xiang and Feng, Jiashi},
  journal={arXiv preprint arXiv:1902.09104},
  year={2019}
}
