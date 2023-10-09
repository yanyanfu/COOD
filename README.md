# COOD: Towards More Trustworthy ML Models for Code: Detecting Out-of-Distribution via Multi-modal Contrastive Learning

## Introduction
This is the official codebase for the paper "Towards More Trustworthy ML Models for Code: Detecting Out-of-Distribution via Multi-modal Contrastive Learning"

## Training Prerequisites
- CUDA 11.0
- python 3.7
- pytorch 1.7.1
- torchvision 0.8.2

## Reproduce the results

```bash
git clone https://anonymous.4open.science/r/COOD-4EA6
cd COOD
python run.py --train
python run.py --test_baseline_metrics
```