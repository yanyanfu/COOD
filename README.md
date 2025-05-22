# Towards More Trustworthy Deep Code Models by Enabling Out-of-Distribution Detection

## Introduction
In this work, We proposed two multi-modal OOD detection methods for code-related pretrained ML models; namely unsupervised COOD and weakly-supervised COOD+. The COOD merely leveraged unsupervised contrastive learning to identify OOD samples. As an extension of COOD, COOD+ combined contrastive learning and a binary classifier for OOD detection using a small number of labelled OOD samples. To reap the benefits of these two modules, we also devised a new scoring metric to fuse their prediction results. 

## Dataset Statistics
The dataset statistics for training and evaluation of our weakly-supervised COOD+ and baseline models are as follows:

<img src="figs/dataset.png" alt="visual results" width="480">


## Training Prerequisites
- python 3.9
- pytorch 1.8.0
- torchvision 0.9.0

## Dataset & Evaluation benchmark

To acquire the training and testing datasets, please first download the datasets used for OOD data generation [here](https://drive.google.com/drive/folders/1GYwQs4klceKFV5c50-G_gxC1EYKtb1sr?usp=sharing), including the datasets constructed from StackOverflow for code search and the datasets constructed from CSN by injecting variable misuse bugs (single token-based). Then, you can run the preprocess.py and preprocess.ipynb under each subfolder of the data folder. Consequently, the OOD data would be extracted automatically based on the dataloader.py file when you train and evaluate our models.

## Reproduce the results

The parameter values in config_java.yaml and config_python.yaml can be changed to adapt to different settings. Our trained COOD and COOD+ model checkpoints are available [here](https://drive.google.com/drive/folders/1KLjtmiCNdPHqU_5wzz9FqZoRxfIv0JfO?usp=drive_link) anonymously for results reproduction.


```bash
git clone https://github.com/yanyanfu/COOD.git
cd COOD
pip install -r requirements.txt
```

To train the model
```bash
python run.py --config config_java.yaml
```

To test the model under OOD detection setting
```bash
python run.py --config config_java.yaml --test_baseline_metrics
```

To test the model on the main code understanding task (i.e. code search) with COOD+ auxiliary
```bash
python run.py --config config_java.yaml  --test_main_task
```
