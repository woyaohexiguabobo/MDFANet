# Enhanced Medical Image Segmentation via Multi-Branch Deformable Convolutional Networks

**Notice:** This code repository is the official implementation of the manuscript submitted to *The Visual Computer* journal.

Please run to get the code:
```shell
git clone --recursive https://github.com/woyaohexiguabobo/MDFANet.git
```

## 1. Overview

### Papers

This repository provides code for our paper "Enhanced Medical Image Segmentation via Multi-Branch Deformable Convolutional Networks"


### Abstract
<p align = "justify">
Medical image segmentation plays a pivotal role in computer-aided diagnosis and treatment planning. Traditional methods often struggle with complex anatomical structures and large-scale variations. This paper introduces MDFANet, a novel U-shaped segmentation model based on deformable convolution and adaptive multi-scale feature aggregation. The model features a six-layer architecture with an enlarged receptive field in the bottleneck, utilizing Deformable Convolutional Residual blocks to mitigate gradient vanishing. An Adaptive Feature Aggregation module dynamically adjusts multi-scale feature importance, while a Multi-Branch Deformable Convolution Fusion module with deep supervision reduces feature misalignment. Experiments on five public datasets demonstrate MDFANet's competitive segmentation performance, achieving a mean Dice score of 92.79% on the 2018DSB dataset and 91.79% on the ISIC 2017 dataset, showcasing its effectiveness in handling complex medical image segmentation tasks.
</p>

## 2. Data Files

### 2.1 Datasets

In this subsection, we provide the public data set used in the paper:

- **Polyp Datasets (Kvasir-SEG & CVC-ClinicDB)**: [Official Website](https://datasets.simula.no/kvasir-seg/)
- **2018 Data Science Bowl**: [Kaggle Official](https://www.kaggle.com/c/data-science-bowl-2018/data)
- **ISIC 2018**: [ISIC Archive](https://challenge.isic-archive.com/data/)
- **ISIC 2017**: [ISIC Archive](https://challenge.isic-archive.com/data/)

## 3. How to run

### 3.1 Create Environment

First of all, you need to have a `pytorch` environment, I use `pytorch 1.10`, but it should be possible to use a lower version, so you can decide for yourself.

You can also create a virtual environment using the following command (note: this virtual environment is named `pytorch`, if you already have a virtual environment with this name on your system, you will need to change `environment.yml` manually).

```shell

```
conda env create -f docs/environment.yml
### 3.2 Training

You can run the following command directly:

```shell
sh run.sh ### use stepLR

```

If you only want to run a single dataset, you can comment out the irrelevant parts of the `sh` file, or just type something like the following command from the command line:

```shell
python Train.py --model_name MDFANet --epoch 151 --batchsize 16 --trainsize 256 --train_save MDFANet_Kvasir_1e4_bs16_e150_s256 --lr 0.0001 --train_path /root/autodl-tmp/MDFANet/data/TrainDataset --test_path /root/autodl-tmp/MDFANet/data/TestDataset/Kvasir/
```

### 3.3 Get all predicted results (.png)

If you use a `sh` file for training, it will be tested after the training is complete.

If you use the `python` command for training, you can also comment out the training part of the `sh` file, or just type something like the following command at the command line:

```shell
python Test.py --train_save MDFANet_Kvasir_1e4_bs16_e150_s256 --testsize 256 --test_path /root/autodl-tmp/MDFANet/data/TestDataset
```

### 3.4 Evaluating

- For evaluating the polyp dataset, you can use the `matlab` code in `eval` or use the evaluation code provided by \[[UACANet](https://github.com/plemeri/UACANet)\].
- For other datasets, you can use the code in [evaldata](https://github.com/z872845991/evaldata/).
- The reason for using a different evaluation code is to use the same methodology in the evaluation as other papers that did experiments on the dataset.




