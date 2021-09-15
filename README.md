# Unified Multi-level Optimization Paradigm
This repo hosts the code for implementing **Unified Multi-level Optimization Paradigm** (UMOP).

## Introduction
In object detection, multi-level prediction (e.g., FPN, YOLO) and resampling skills (e.g., focal loss, ATSS) have drastically improved one-stage detector performance. However, how to improve the performance by optimizing the feature pyramid level-by-level remains unexplored. We find that, during training, the ratio of positive over negative samples varies across pyramid levels (**level imbalance**), which is not addressed by current one-stage detectors. To mediate the influence of level imbalance, we propose a Unified Multi-level Optimization Paradigm (UMOP) consisting of two components: 1) an independent classification loss supervising each pyramid level with individual resampling considerations; 2) a progressive hard-case mining loss defining all losses across the pyramid levels without extra level-wise settings. With UMOP as a plug-and-play scheme, modern one-stage detectors can attain a **~1.5 AP** improvement with fewer training iterations and no additional computation overhead. Our best model achieves **55.1** AP on COCO `test-dev`. 

## Updates
- **2021.09.24** This repo has been commited, based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/a1cecf63c713c53941b8dcf8a9d762baf8511f2c) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

## Installation
The same installation procedure in MMDetection and Swin-Transformer-Object-Detection should be kept, including the **mmcv**, **apex**, and **timm**.

## Main Results
For your convenience, we provide the following trained models. These models are trained with a mini-batch size of 16 images on 8 Nvidia V100 GPUs (2 images per GPU), except that the largest backbone is trained on 8 Nvidia P40 GPUs with [apex](https://github.com/NVIDIA/apex).

| Backbone     | Style     | DCN     | MS <br> train | MS <br> test |Lr <br> schd | box AP <br> (val) | box AP <br> (test-dev) |
|:------------:|:---------:|:-------:|:-------------:|:------------:|:------------------:|:-----------------:|:----------------------:|
| R-50         | pytorch   | N       | N             | N             | 1.5x           | 40.4              | -                   |
| R-101        | pytorch   | N       | N             | N             | 1.5x           | 42.1              | 42.3                   |
| R-101        | pytorch   | Y       | N             | N             | 1.5x           | 45.2              | 45.4                   |
| R-101        | pytorch   | Y       | Y             | N             | 1.5x           | 47.6              | 47.7                   |
| X-101-64x4d  | pytorch   | Y       | Y             | N             | 1.5x           | 48.8              | 49.1                   |
| R2-101       | pytorch   | Y       | Y             | N             | 2x           | 50.0              | 50.3                   |
| Swin-S       | pytorch   | N       | Y             | N             | 2x           | 49.9              | 50.3                   |
| Swin-S       | pytorch   | N       | Y             | Y             | 2x           | 51.9              | 52.3                   |
| Swin-B       | pytorch   | N       | Y             | N             | 2x           | 51.6              | 51.9                   |
| Swin-B       | pytorch   | N       | Y             | Y             | 2x           | 53.4              | 53.9                   |
| Swin-L       | pytorch   | N       | Y             | N             | 2x           | 52.8              | 53.1                   |
| Swin-L       | pytorch   | N       | Y             | Y             | 2x           | 54.7              | 55.1                   |
