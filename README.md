# Unified Multi-level Optimization Paradigm
This repo hosts the code for implementing [UMOP](https://arxiv.org/abs/2109.07217) (**Unified Multi-level Optimization Paradigm**).

## Introduction
In object detection, multi-level prediction (e.g., FPN, YOLO) and resampling skills (e.g., focal loss, ATSS) have drastically improved one-stage detector performance. However, how to improve the performance by optimizing the feature pyramid level-by-level remains unexplored. We find that, during training, the ratio of positive over negative samples varies across pyramid levels **(Level Imbalance)**, which is not addressed by current one-stage detectors. 

To mediate the influence of level imbalance, we propose a Unified Multi-level Optimization Paradigm (UMOP) consisting of two components: 1) an independent classification loss supervising each pyramid level with individual resampling considerations; 2) a progressive hard-case mining loss defining all losses across the pyramid levels without extra level-wise settings. With UMOP as a plug-and-play scheme, modern one-stage detectors can attain a **~1.5 AP** improvement with fewer training iterations and no additional computation overhead. Our best model achieves **55.1** AP on COCO test-dev. 

## Updates
- **2021.09.17** We committed the coco pretrain model and train logs.
- **2021.09.14** This repo has been committed, based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/a1cecf63c713c53941b8dcf8a9d762baf8511f2c) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

## Installation
- The installation is the same as [MMDetection](https://github.com/open-mmlab/mmdetection/tree/a1cecf63c713c53941b8dcf8a9d762baf8511f2c) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).
- Please check [get_started.md](docs/get_started.md) for installation, our recommended installation is `Pytorch=1.7.1, torcivision=0.8.2, mmcv=1.3.11`
- Please install `timm` and `apex` for swin backbones, `timm=0.4.12` is recommended.
- To install `apex`, please run:
  ```
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```
- The pretrained `Swin` backbone could be obtained at [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also provide a third party pack ([Google Drive](https://drive.google.com/file/d/1b8VXGqI2TRkOSjueDr1sQNb6eBMAiUc0/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1u3a8yw3mNcjzfu0xEVGQNg) with access code `io5i`). Please download and unzip it.
- If you download from the official link, please move these models according to the config file (**Maybe you should rename them**).

## Main Results
For your convenience, we provide the following trained models. These models are trained with a mini-batch size of 16 images on 8 Nvidia V100 GPUs (2 images per GPU), except that the largest backbone (Swin-L) is trained on 8 Nvidia P40 GPUs with [apex](https://github.com/NVIDIA/apex).

| Backbone     | DCN | MS <br> train | MS <br> test |Lr <br> schd | box AP <br> (val) | box AP <br> (test-dev) | Download |
|:------------:|:---:|:--------:|:-------:|:------:|:------------:|:-----------------:|:--------:|
| R-50         | N   | N        | N       | 1.5x   | 40.4         | 40.5              | [model](https://drive.google.com/file/d/1sP1C6BRQub_NMCZlQm1AYUt2fv_B9UMp/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ndLSgjs9NaUb4IQ63wvNy-Q23ESDWzMP/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1roc5lQTEjwyALew2RkO36TMUJzl39ey8/view?usp=sharing)|
| R-101        | N   | N        | N       | 1.5x   | 42.1         | 42.3              | [model](https://drive.google.com/file/d/1GtB9e25Dpu8qtkrozV8eHDB5Lxvtq2PW/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1wxybC-3worV6L2nru9YzGkHF-gO7LtPe/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1XApYHxuimErvFQdkduSCsflBQpkSpGET/view?usp=sharing)|
| R-101        | Y   | N        | N       | 1.5x   | 45.2         | 45.4              | [model](https://drive.google.com/file/d/1dtJTiBNGhA-Z_OgXwmfWynbHssvYC49-/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1SUJdVxl3k6d220u5kYoW7Z_FmZVnFRbi/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1TWlk9v41QPH7uEJja2wjnnz_9azoj3vx/view?usp=sharing)|
| R-101        | Y   | Y        | N       | 1.5x   | 47.6         | 47.7              | [model](https://drive.google.com/file/d/1f2QBWs5evb5_xtOmT8B4W9aH1aWO0bkZ/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pjXhSAnnjOucqTL602I3vOPH0CXNaPnb/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1fM6ZGXsPetWGfGyIKpLvZwTTHKTXRzuu/view?usp=sharing)|
| X-101-64x4d  | Y   | Y        | N       | 1.5x   | 48.8         | 49.1              | [model](https://drive.google.com/file/d/1LLsFCzXPywhiwyhokeq9ltdgZVbVRlKN/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/10eJjyl5rtZ-TeNUjCk0BYclqXNBvLxEz/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1BC51Q27B0RpX3qnqDHNYyeX42N2sxTix/view?usp=sharing)|
| R2-101       | Y   | Y        | N       | 2x     | 50.0         | 50.3              | [model](https://drive.google.com/file/d/1AvFiwi4k-VpG3NjRY29gWz8Q9Lm0zxZb/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/12vx4RzZ-uPQll76zGnE4Mr05Wzb_DrEh/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1hMJl4qK2TQdw_mk4qRfGckcZ52V5jR-z/view?usp=sharing)|
| Swin-S       | N   | Y        | N       | 2x     | 49.9         | 50.3              | [model](https://drive.google.com/file/d/1e4SZ7K_PQUEf0bLBZWJ6BmtOJFt6Mp-V/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/17qQWWsY8a5Vi3dmDWXwSFHJByUOzze0c/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1gvbZgj2wVsGJq51OXc0qRuGPV74KvvmP/view?usp=sharing)|
| Swin-S       | N   | Y        | Y       | 2x     | 51.9         | 52.3              | [model](https://drive.google.com/file/d/1e4SZ7K_PQUEf0bLBZWJ6BmtOJFt6Mp-V/view?usp=sharing) &#124; - &#124; [JSON](https://drive.google.com/file/d/1fCRTX1x7D6KIlRqNRe6FZnT_Gspx3t1b/view?usp=sharing)|
| Swin-B       | N   | Y        | N       | 2x     | 51.6         | 51.9              | [model](https://drive.google.com/file/d/1mmKUNN81VSwEVmRQuIBgKxrOsl_8AadG/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1xzIPr5trn9AmpAbnNTtfc4T5rfwBSDqM/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1hZ_n2JIqxCl0ARWt7I5XQ4_OyWYMrcyd/view?usp=sharing)|
| Swin-B       | N   | Y        | Y       | 2x     | 53.4         | 53.9              | [model](https://drive.google.com/file/d/1mmKUNN81VSwEVmRQuIBgKxrOsl_8AadG/view?usp=sharing) &#124; - &#124; [JSON](https://drive.google.com/file/d/1YS4BPjkNfh2vS_nSEADbq3a2v7b8Ig9a/view?usp=sharing)|
| Swin-L       | N   | Y        | N       | 2x     | 52.8         | 53.1              | [model](https://drive.google.com/file/d/1xnUz4Jlz2NFgFNHLnj3Am7K6d5hfjpaG/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/17p-I0WZbkh1TULY7pn8joUjt10CtntSL/view?usp=sharing) &#124; [JSON](https://drive.google.com/file/d/1p4Whr81dJ1rUB2UBwyM_fZOBWmfCGWXs/view?usp=sharing)|
| Swin-L       | N   | Y        | Y       | 2x     | 54.7         | 55.1              | [model](https://drive.google.com/file/d/1xnUz4Jlz2NFgFNHLnj3Am7K6d5hfjpaG/view?usp=sharing) &#124; - &#124; [JSON](https://drive.google.com/file/d/1T3wPYi9zesSq7A9s0TRqv_LL5zHjzNFp/view?usp=sharing)|
