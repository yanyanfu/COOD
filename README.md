# COOD: Towards More Trustworthy ML Models for Code: Detecting Out-of-Distribution via Multi-modal Contrastive Learning

## Introduction
This is the official codebase for the paper "Towards More Trustworthy ML Models for Code: Detecting Out-of-Distribution via Multi-modal Contrastive Learning". In this work, we propose a general-purpose multi-modal OOD detection framework for code related pretrained
models that combines contrastive learning and a binary classifier. To achieve this, we employed a
margin-based loss in contrastive learning to maximize the difference in similarity scores between ID
and OOD code pairs. In addition, we introduced a binary OOD rejection network that can be adapted
to be used in conjunction with contrastive learning for further enhancing detection performance. The
evaluation results demonstrated that the integration of the rejection network and contrastive
learning can achieve superior performance in detecting all four OOD scenarios for multi-modal
NL-PL data. Additionally, our model can be applied to the downstream code understanding task, namely code search, achieving comparable
performance to several recent pretrained models.

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