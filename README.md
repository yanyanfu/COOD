# Towards More Trustworthy Deep Code Models by Enabling Out-of-Distribution Detection

## Introduction
This is the official codebase for the paper "Towards More Trustworthy Deep Code Models by Enabling Out-of-Distribution Detection". In this work, We proposed two multi-modal OOD detection methods for code related pretrained ML models; namely unsupervised COOD and weakly-supervised COOD+. The COOD merely leveraged unsupervised contrastive learning to identify OOD samples. As an extension of COOD, COOD+ combined contrastive learning and a binary classifier for OOD detection using a small number of labelled OOD samples. To reap the benefits of these two modules, we also devised a new scoring metric to fuse their prediction results. The evaluation results demonstrated that the integration of the rejection network and contrastive learning can achieve superior performance in detecting all four OOD scenarios for multi-modal NL-PL data. Additionally, our models can be applied to the downstream task, achieving comparable performance to several current pre-trained models.

## Training Prerequisites
- CUDA 11.5
- python 3.9
- pytorch 1.8.0
- torchvision 0.9.0

## Dataset & Evaluation benchmark

To acquire the training and testing datasets, please run the preprocess.py and preprocess.ipynb under each subfolder of the data folder. Then, the OOD data would be extracted automatically based on the dataloader.py file when you train and evaluate the models.

## Reproduce the results

The parameter values in config_java.yaml and config_python.yaml can be changed to adapt to different settings.

```bash
git clone https://anonymous.4open.science/r/COOD-4EA6
cd COOD
python run.py --train
python run.py --test_baseline_metrics
python run.py --test_main_task
```