import numpy as np
import os
import shutil
import copy
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.optim import AdamW, SGD
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from data.data_loader import AnomalyDatasetWrapper
from model.model import AnomalyDetectionModel
from model.loss import ContrastiveLoss, ClassificationLoss
from torchsummary import summary

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def plot_distribution(test_score, threshold_score, test_label, img_path):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 100)
    ax.hist(test_score[test_label >= 0], bins=bins, alpha=0.5, color='#7f7f7f', align='mid', label='ID')
    ax.hist(test_score[test_label == -1], bins=bins, alpha=0.5, color='#1f77b4', align='mid', label='Scenario 1')
    ax.hist(test_score[test_label == -3], bins=bins, alpha=0.5, color='#ff7f0e', align='mid', label='Scenario 2')
    ax.hist(test_score[test_label == -4], bins=bins, alpha=0.5, color='#2ca02c', align='mid', label='Scenario 3')
    # ax.hist(test_score[test_label == -4], bins=bins, alpha=0.5, color='#e6b8af', align='mid', label='Scenario 4')
    ax.axvline(x=threshold_score, color='k', linestyle='--', linewidth=2, label='Threshold: {:.2f}'.format(threshold_score))
    ax.set_xlim([0.0, 1.0])
    ax.set_xticks(0.1 * np.arange(0, 11), fontsize=15)
    ax.set_xticklabels(['{:.1f}'.format(0.1 * i) for i in np.arange(0, 11)], fontsize=15)
    ax.set_ylim([0, 140])
    ax.set_yticks(20 * np.arange(0, 8), fontsize=15)
    ax.set_yticklabels([str(20 * i) for i in np.arange(0, 8)], fontsize=15)
    ax.legend(loc="upper left", fontsize=15)
    plt.savefig(img_path)
    plt.close()


def plot_aucroc(test_anom, test_auc, test_label, img_path):
    fpr, tpr, thresholds = roc_curve(np.where(test_label < 0, 0, 1), test_anom)
    df_tpr_fpr = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds})
    gmean = np.sqrt(tpr * (1 - fpr))
    thresholds_opt = round(thresholds[np.argmax(gmean)], 4)
    gmean_opt = round(gmean[np.argmax(gmean)], 4)
    tpr_opt = round(tpr[np.argmax(gmean)], 4)
    fpr_opt = round(fpr[np.argmax(gmean)], 4)
    # print(thresholds)
    plt.clf()
    fig, ax = plt.subplots()
    fpr95 = fpr_and_fdr_at_recall(np.where(test_label < 0, 0, 1), test_anom)
    ax.plot(fpr, tpr, label='AUCROC: {:.2f}\n'.format(test_auc * 100), color='#1f77b4', linewidth=2)
    ax.scatter(fpr95, 0.95, color='#ff7f0e', label='FPR95: {:.2f}'.format(fpr95 * 100))
    
    ax.set_xlabel('False Positive Rate (FPR))', fontsize=15)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=15)
    ax.legend(loc="lower right", fontsize=15)
    fig.savefig(img_path)
    plt.close()


class AnomalyTrainer(object):
    def __init__(self, 
                 rank,
                 config,
                 seed):

        self.seed = seed

        config_file = config
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if config['mode'] == 'unsup':
            log_dir = '{}_unsup_seed{}'.format(config['dataset']['name'], 0)
        elif config['mode'] == 'wsup':
            log_dir = '{}_m{}_cl{}_bc{}_seed{}_nogg'.format(config['dataset']['name'], config['margin'], config['lambda_cl'], config['lambda_bc'], 0)

        self.writer = SummaryWriter(log_dir=os.path.join('runs/graphcodebert', log_dir))
        self.model_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        os.makedirs(self.model_folder, exist_ok=True)
        shutil.copy(config_file, os.path.join(self.writer.log_dir, 'config.yaml'))

        self.dataset = AnomalyDatasetWrapper(**config['dataset'])
        self.model = AnomalyDetectionModel(**config['model'])
        print('Model summary:')
        summary(self.model)

        self.mode = config['mode']
        self.task = config['task']
        self.loss_cl = ContrastiveLoss(config['contrastive_loss'], task=self.task, margin=config['margin'])
        self.loss_bc = ClassificationLoss(task=self.task)
        
        self.gpu_ids = config['gpu_ids']
        self.world_size = len(self.gpu_ids)
        self.rank = rank

        self.test_dataset = config['test_dataset']
        self.learning_rate = config['learning_rate']
        self.n_epochs = config['n_epochs']
        self.anom_score = config['anom_score']
        self.threshold = config['threshold']
        self.temperature = config['temperature']
        self.lambda_bc = config['lambda_bc']
        self.lambda_cl = config['lambda_cl']

    def _adjust_learning_rate(self, optimizer, epoch, factor=10):
        lr = self.learning_rate
        if epoch > 5:
            lr /= factor
        if epoch > 10:
            lr /= factor
        if epoch > 20:
            lr /= factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _contrastive_score(self, contrastive_features):
        code_features, text_features = contrastive_features
        similarity = code_features @ text_features.t() / self.temperature
        if self.anom_score == 'softmax':
            score = torch.sigmoid(similarity.diag())
        else:
            score = similarity.diag()
        return score

    def _evaluate(self, loader, device):
        self.model.eval()
        with torch.no_grad():
            total_loss_cl = 0.0
            total_loss_bc = 0.0
            total_prob = torch.tensor([], dtype=torch.float).to(device)
            total_score = torch.tensor([], dtype=torch.float).to(device)
            total_label = torch.tensor([], dtype=torch.long).to(device)
            total_nl_vec = torch.tensor([], dtype=torch.float).to(device)
            total_pl_vec = torch.tensor([], dtype=torch.float).to(device)

            for (code, text, label) in tqdm(loader):
                code = code.to(device)
                text = text.to(device)
                label = label.to(device)

                contrastive_features, mm_features, anom_output = self.model(code, text)
                loss_cl = self.loss_cl(contrastive_features, label=label)
                loss_bc = self.loss_bc(mm_features, anom_output, label=label)

                if self.mode == 'wsup':
                    score = self._contrastive_score(contrastive_features)
                    total_score = torch.cat((total_score, score), dim=0)
                    prob = torch.sigmoid(anom_output)
                    total_prob = torch.cat((total_prob, prob), dim=0)

                pl_vec, nl_vec = contrastive_features
                total_nl_vec = torch.cat((total_nl_vec, nl_vec), dim=0)
                total_pl_vec = torch.cat((total_pl_vec, pl_vec), dim=0)
                    
                total_label = torch.cat((total_label, label), dim=0)
                total_loss_cl += loss_cl.item()
                total_loss_bc += loss_bc.item()

        total_loss_cl /= len(loader)
        total_loss_bc /= len(loader)
        total_label = total_label.detach().cpu().numpy()
        total_score = total_score.detach().cpu().numpy()
        total_prob = total_prob.detach().cpu().numpy()
        total_nl_vec = total_nl_vec
        total_pl_vec = total_pl_vec
        
        return total_loss_cl, total_loss_bc, total_label, total_score, total_prob, total_nl_vec, total_pl_vec
        
    def train(self):
        train_loader = self.dataset.get_train_loaders()
        valid_loader = self.dataset.get_train_loaders(dataset_load = 'valid')

        device = torch.device("cuda:{}".format(self.rank) if torch.cuda.is_available() else "cpu")
        self.train_folder = os.path.join(self.writer.log_dir, 'train')
        os.makedirs(self.train_folder, exist_ok=True)
        self.model.to(device)
        self.loss_cl.to(device)
        self.loss_bc.to(device)

        if self.world_size > 1:
            self.model = DP(self.model, device_ids=self.gpu_ids)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.2, betas=(0.9, 0.999))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader)*self.n_epochs*0.1, num_training_steps=len(train_loader)*self.n_epochs)
        best_valid_auc = -np.inf
        best_valid_mrr = -np.inf

        print('Start training...')
        for i in range(self.n_epochs):
            train_loss_cl = 0.0
            train_loss_bc = 0.0
            train_loss = 0.0
            train_label = torch.tensor([], dtype=torch.long).to(device)
            train_score = torch.tensor([], dtype=torch.float).to(device)
            train_prob = torch.tensor([], dtype=torch.float).to(device)

            self.model.train()
            for (code, text, label) in tqdm(train_loader):
                code = code.to(device)
                text = text.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                contrastive_features, mm_features, anom_output = self.model(code, text)
                loss_cl = self.loss_cl(contrastive_features, label=label)
                loss_bc = self.loss_bc(mm_features, anom_output, label=label)

                loss = self.lambda_cl * loss_cl + self.lambda_bc * loss_bc
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_cl += loss_cl.item()
                train_loss_bc += loss_bc.item()
                train_loss += loss.item()
                
                if self.mode == 'wsup':
                    score = self._contrastive_score(contrastive_features)
                    train_score = torch.cat((train_score, score), dim=0)
                    prob = torch.sigmoid(anom_output)
                    train_prob = torch.cat((train_prob, prob), dim=0)
                
                train_label = torch.cat((train_label, label), dim=0)

            train_loss_cl /= len(train_loader)
            train_loss_bc /= len(train_loader)
            train_loss /= len(train_loader)

            self.writer.add_scalar('train/loss_cl', train_loss_cl, i)
            self.writer.add_scalar('train/loss_bc', train_loss_bc, i)
            self.writer.add_scalar('train/loss', train_loss, i)
            self.writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], i)
            print('Epoch: {}\tTrain Loss: {:.4f}'.format(i, train_loss))

            train_label = train_label.detach().cpu().numpy()
            train_score = train_score.detach().cpu().numpy()
            train_prob = train_prob.detach().cpu().numpy()
            
            if self.mode == 'wsup':
                train_anom = train_score * train_prob
                # train_anom = train_score
                train_auc = roc_auc_score(np.where(train_label < 0, 0, 1), train_anom)
                train_auc_cl = roc_auc_score(np.where(train_label < 0, 0, 1), train_score)
                train_auc_bc = roc_auc_score(np.where(train_label < 0, 0, 1), train_prob)
                self.writer.add_scalar('train/auc_cl', train_auc_cl, i)
                self.writer.add_scalar('train/auc_bc', train_auc_bc, i)
                self.writer.add_scalar('train/auc', train_auc, i)
                print('Epoch: {}\tTrain AUC (CL): {:.4f}\tTrain AUC (BC): {:.4f}\tTrain AUC: {:.4f}'.format(i, train_auc_cl, train_auc_bc, train_auc))

            valid_loss_cl, valid_loss_bc, valid_label, valid_score, valid_prob, valid_nl_vec, valid_pl_vec = self._evaluate(valid_loader, device)

            valid_loss = valid_loss_cl * self.lambda_cl + valid_loss_bc * self.lambda_bc
            self.writer.add_scalar('valid/loss_cl', valid_loss_cl, i)
            self.writer.add_scalar('valid/loss_bc', valid_loss_bc, i)
            self.writer.add_scalar('valid/loss', valid_loss, i)
            print('Epoch: {}\tValid Loss: {:.4f}'.format(i, valid_loss))

            if self.mode == 'wsup':
                valid_anom = valid_score * valid_prob
                # valid_anom = valid_score
                valid_auc_cl = roc_auc_score(np.where(valid_label < 0, 0, 1), valid_score)
                valid_auc_bc = roc_auc_score(np.where(valid_label < 0, 0, 1), valid_prob)
                valid_auc = roc_auc_score(np.where(valid_label < 0, 0, 1), valid_anom)
                self.writer.add_scalar('valid/auc_cl', valid_auc_cl, i)
                self.writer.add_scalar('valid/auc_bc', valid_auc_bc, i)
                self.writer.add_scalar('valid/auc', valid_auc, i)
                print('Epoch: {}\tValid AUC (CL): {:.4f}\tValid AUC (BC): {:.4f} \tValid AUC: {:.4f}'.format(i, valid_auc_cl, valid_auc_bc, valid_auc))
        
            valid_nl_vec = valid_nl_vec[valid_label >= 0]
            valid_pl_vec = valid_pl_vec[valid_label >= 0]
            valid_scores = valid_nl_vec @ valid_pl_vec.T
            correct_scores = torch.diag(valid_scores)
            compare_scores = valid_scores >= correct_scores.unsqueeze(1)
            valid_ranks = 1.0 / compare_scores.float().sum(1)
            valid_mrr = valid_ranks.mean()
            
            self.writer.add_scalar('valid/mrr', valid_mrr, i)
            print('Epoch: {}\tValid MRR: {:.4f}'.format(i, valid_mrr))

            if self.mode == 'unsup':
                if valid_mrr > best_valid_mrr:
                    best_valid_mrr = valid_mrr
                    torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'best_model.pt'))
                    print('Best model saved.')
            else:
                if not self.lambda_cl:
                    valid_auc = valid_auc_bc
                elif not self.lambda_bc:
                    valid_auc = valid_auc_cl
                if (valid_auc + valid_mrr) > (best_valid_auc + best_valid_mrr):
                    best_valid_auc = valid_auc
                    best_valid_mrr = valid_mrr
                    torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'best_model.pt'))
                    print('Best model saved.')


            if i % 5 == 0 or i == self.n_epochs - 1:
                torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'model_epoch{}.pt'.format(i)))
                print('Model saved.')

        self.writer.flush()
        print('Finished training')

    
    def test_baseline_metrics(self, model_path=None):
        anom_dict = {0: 'ID', 1: 'out-domain', 3: 'wrong-text', 2: 'shuffle', 4: 'buggy'}
        test_loader = self.dataset.get_test_loaders(self.test_dataset, dataset_load = 'test')
        device = torch.device('cuda:{}'.format(self.rank) if torch.cuda.is_available() else torch.device('cpu'))
        self.test_folder = os.path.join(self.writer.log_dir, 'test_base', str(self.seed))
        os.makedirs(self.test_folder, exist_ok=True)

        if self.world_size > 1:
            self.model = DP(self.model, device_ids=range(self.world_size))

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.model_folder, 'best_model.pt')))
        self.model.to(device)
        
        _, _, test_label, test_score, test_prob, test_nl_vec, test_pl_vec = self._evaluate(test_loader, device)
        test_scores = (test_nl_vec @ test_pl_vec.T).detach().cpu().numpy()
        
        if self.dataset.anom_test > 0.0:
            if self.mode == 'wsup':
                test_anom = test_score * test_prob               
                for i in range(1, 5):
                    print('Scenario {}: {}'.format(i, anom_dict[i]))
                    result = pd.DataFrame(columns=['auc-roc', 'fpr95', 'aupr-ID', 'aupr-OOD'])
                    scene = np.where((test_label >= 0) | (test_label == -i))[0]
                    label = np.where(test_label[scene] < 0, 0, 1)
                    OOD_label = np.where(test_label[scene] < 0, 1, 0)
                    result.loc['Contrastive'] = [roc_auc_score(label, test_score[scene]), fpr_and_fdr_at_recall(label, test_score[scene]), average_precision_score(label, test_score[scene]), average_precision_score(OOD_label, [-x for x in test_score[scene]])]
                    result.loc['Classification'] = [roc_auc_score(label, test_prob[scene]), fpr_and_fdr_at_recall(label, test_prob[scene]), average_precision_score(label, test_prob[scene]), average_precision_score(OOD_label, [-x for x in test_prob[scene]])]
                    result.loc['Anomaly'] = [roc_auc_score(label, test_anom[scene]), fpr_and_fdr_at_recall(label, test_anom[scene]), average_precision_score(label, test_anom[scene]), average_precision_score(OOD_label, [-x for x in test_anom[scene]])]

                    result.round(4).to_csv(os.path.join(self.test_folder, 'result_scenario_auc_{}.csv'.format(i)))
                    print('=====================================')


                print('All Scenarios')
                result = pd.DataFrame(columns=['auc-roc', 'fpr95', 'aupr-ID', 'aupr-OOD'])

                label = np.where(test_label < 0, 0, 1)
                OOD_label = np.where(test_label < 0, 1, 0)
                OOD_test_score = [-x for x in test_score]
                OOD_test_prob = [-x for x in test_prob]
                OOD_test_anom = [-x for x in test_anom]

                result.loc['Contrastive'] = [roc_auc_score(label, test_score), fpr_and_fdr_at_recall(label, test_score), average_precision_score(label, test_score), average_precision_score(OOD_label, OOD_test_score)]
                result.loc['Classification'] = [roc_auc_score(label, test_prob), fpr_and_fdr_at_recall(label, test_prob), average_precision_score(label, test_prob), average_precision_score(OOD_label, OOD_test_prob)]
                result.loc['Anomaly'] = [roc_auc_score(label, test_anom), fpr_and_fdr_at_recall(label, test_anom), average_precision_score(label, test_anom), average_precision_score(OOD_label, OOD_test_anom)]
                result.round(4).to_csv(os.path.join(self.test_folder, 'result_all_scenarios_auc.csv'))

            
            elif self.mode == 'unsup':
                test_anom = np.array([test_scores[i, i] for i in range(len(test_scores))])

                for i in range(1, 5):
                    print('Scenario {}: {}'.format(i, anom_dict[i]))
                    result = pd.DataFrame(columns=['auc-roc', 'fpr95', 'aupr-ID', 'aupr-OOD'])
                    scene = np.where((test_label >= 0) | (test_label == -i))[0]
                    label = np.where(test_label[scene] < 0, 0, 1)
                    OOD_label = np.where(test_label[scene] < 0, 1, 0)
                    result.loc['Anomaly'] = [roc_auc_score(label, test_anom[scene]), fpr_and_fdr_at_recall(label, test_anom[scene]), average_precision_score(label, test_anom[scene]), average_precision_score(OOD_label, [-x for x in test_anom[scene]])]

                    result.round(4).to_csv(os.path.join(self.test_folder, 'result_scenario_auc_{}.csv'.format(i)))
                    print('=====================================')


                print('All Scenarios')
                result = pd.DataFrame(columns=['auc-roc', 'fpr95', 'aupr-ID', 'aupr-OOD'])
                label = np.where(test_label < 0, 0, 1)
                OOD_label = np.where(test_label < 0, 1, 0)
                OOD_test_anom = [-x for x in test_anom]
                result.loc['Anomaly'] = [roc_auc_score(label, test_anom), fpr_and_fdr_at_recall(label, test_anom), average_precision_score(label, test_anom), average_precision_score(OOD_label, OOD_test_anom)]
                result.round(4).to_csv(os.path.join(self.test_folder, 'result_all_scenarios_auc.csv'))

                
        print('Finished testing')


    def test_main_task(self, model_path=None):
        anom_dict = {0: 'ID', 1: 'out-domain', 3: 'wrong-text', 4: 'buggy'}
        test_loader = self.dataset.get_test_loaders(self.test_dataset, test_type='test')
        codebase_loader = self.dataset.get_test_loaders(self.test_dataset, dataset_load = 'codebase')

        device = torch.device('cuda:{}'.format(self.rank) if torch.cuda.is_available() else torch.device('cpu'))
        self.test_folder = os.path.join(self.writer.log_dir, 'test_main', str(self.seed))
        os.makedirs(self.test_folder, exist_ok=True)

        if self.world_size > 1:
            self.model = DP(self.model, device_ids=range(self.world_size))

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            checkpoint = torch.load(os.path.join(self.model_folder, 'best_model.pt'), map_location='cpu')
            self.model.load_state_dict(checkpoint)
            # self.model.load_state_dict(torch.load(os.path.join(self.model_folder, 'best_model.pt')))
        self.model.to(device)
        
        _, _, test_label, test_score, test_prob, test_nl_vec, test_pl_vec = self._evaluate(test_loader, device)
        _, _, _, _, _, _, test_pl_codebase_vec = self._evaluate(codebase_loader, device)
        test_pl_vec = test_pl_vec.detach().cpu().numpy()
        test_nl_vec = test_nl_vec.detach().cpu().numpy()
        test_pl_codebase_vec = test_pl_codebase_vec.detach().cpu().numpy()
        print(len(test_pl_vec))
        print(len(test_pl_codebase_vec))

        test_scores = np.matmul(test_nl_vec, test_pl_vec.T)
        
        if self.dataset.anom_test > 0.0:
            if self.mode == 'wsup':
                test_anom = test_score * test_prob
                test_auc_cl = roc_auc_score(np.where(test_label < 0, 0, 1), test_score)
                test_auc_bc = roc_auc_score(np.where(test_label < 0, 0, 1), test_prob)
                test_auc = roc_auc_score(np.where(test_label < 0, 0, 1), test_anom)
                print('Test AUC (CL): {:.4f}\tTest AUC (BC): {:.4f}\tTest AUC (Anom): {:.4f}'.format(test_auc_cl, test_auc_bc, test_auc))

                # find threshold so 95% of ID samples are classified as ID
                id_score = sorted(copy.deepcopy(test_score[test_label >= 0]), reverse=True)
                id_prob = sorted(copy.deepcopy(test_prob[test_label >= 0]), reverse=True)
                id_anom = sorted(copy.deepcopy(test_anom[test_label >= 0]), reverse=True)
                id_ratio = 0.95
                threshold_score = id_score[int(id_ratio * len(id_score))]
                threshold_prob = id_prob[int(id_ratio * len(id_prob))]
                threshold_anom = id_anom[int(id_ratio * len(id_anom))]
            
            elif self.mode == 'unsup':
                test_anom = np.array([test_scores[i, i] for i in range(len(test_scores))])
                test_auc = roc_auc_score(np.where(test_label < 0, 0, 1), test_anom)
                print('Test AUC (Anom): {:.4f}'.format(test_auc))
                id_anom = sorted(copy.deepcopy(test_anom[test_label >= 0]), reverse=True)
                id_ratio = 0.95
                threshold_anom = id_anom[int(id_ratio * len(id_anom))]

        test_scores = np.matmul(test_nl_vec, np.vstack((test_pl_vec, test_pl_codebase_vec)).T)
        print(test_scores.shape)

        raw_test_ranks = []
        for i in tqdm(range(len(test_scores))):
            score = test_scores[i, i]
            rank = 1
            for j in range(len(test_scores)):
                if j != i and test_scores[i, j] >= score:
                    rank += 1
            raw_test_ranks.append(1.0 / rank)        
        raw_test_mrr = np.mean(raw_test_ranks)
        print('Raw Test MRR: {:.4f}'.format(raw_test_mrr))

        gold_test_nl_vec = test_nl_vec[test_label >= 0]
        gold_test_pl_vec = test_pl_vec[test_label >= 0]
        gold_test_scores = np.matmul(gold_test_nl_vec, np.vstack((gold_test_pl_vec, test_pl_codebase_vec)).T)
        gold_test_ranks = []
        for i in tqdm(range(len(gold_test_scores))):
            score = gold_test_scores[i, i]
            rank = 1
            for j in range(len(gold_test_scores)):
                if j != i and gold_test_scores[i, j] >= score:
                    rank += 1
            gold_test_ranks.append(1.0 / rank)        
        gold_test_mrr = np.mean(gold_test_ranks)
        print('Gold Test MRR: {:.4f}'.format(gold_test_mrr))

        filtered_test_nl_vec = test_nl_vec[test_anom >= threshold_anom]
        filtered_test_pl_vec = test_pl_vec[test_anom >= threshold_anom]
        filtered_test_scores = np.matmul(filtered_test_nl_vec, np.vstack((filtered_test_pl_vec, test_pl_codebase_vec)).T)
        filtered_test_ranks = []
        for i in tqdm(range(len(filtered_test_scores))):
            score = filtered_test_scores[i, i]
            rank = 1
            for j in range(len(filtered_test_scores)):
                if j != i and filtered_test_scores[i, j] >= score:
                    rank += 1
            filtered_test_ranks.append(1.0 / rank)        
        filtered_test_mrr = np.mean(filtered_test_ranks)
        print('Filtered Test MRR: {:.4f}'.format(filtered_test_mrr))
        
        print('Finished testing main task')
