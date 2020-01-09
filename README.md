# Arithmetic Addition Classification
This repository contains the code (in PyTorch) for "[Arithmetic addition by deep image classification networks: experiments to quantify autonomous reasoning ability](https://arxiv.org/abs/1912.04518)" paper by [Shuaicheng Liu](http://www.liushuaicheng.org/), Zehao Zhang, Kai Song, and Bing Zeng.
## Introduction
In this work, we design a series of experiments, inspired by children’s learning of arithmetic additions of two integers, to showcase that such networks can go beyond the structural features to learn deeper knowledge.
![add image](https://github.com/kaileysong/arithadd/blob/master/CNN_arch.png)
## Requirements
- [Python3.6+](https://www.python.org/downloads/)
- [PyTorch(1.2.0+)](http://pytorch.org)
- torchvision 0.2.0+
- numpy
## Train & Evaluation
1. Use the following command to generate the formula images first.
```
python makeimg.py
```
2. Run the following code to train se_resnet and observe the classification results directly.
```
python train.py
```
3. Results above corresponds to the first experiments validating commutative law by default, you can modify train_val_txt.py to conduct other experiments.
## Citation 
If you use our code or method in your work, please cite the following:
```
@misc{liu2019arithmetic,
    title={Arithmetic addition of two integers by deep image classification networks: experiments to quantify their autonomous reasoning ability},
    author={Shuaicheng Liu and Zehao Zhang and Kai Song and Bing Zeng},
    year={2019},
    eprint={1912.04518},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
