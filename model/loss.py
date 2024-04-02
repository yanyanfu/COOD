import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, mode, task='codesearch', margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.mode = mode
        self.task = task
        self.margin = margin
        
    def forward(self, features, label=None):
        code_features, text_features = features
        if self.mode == "margin":
            sim = (code_features @ text_features.t()).clamp(min=0.0)
            sim_diag = torch.diag(sim).view(code_features.shape[0], 1)
            mask = torch.eye(sim.shape[0], device=sim.device).bool()
            ood_indices = torch.where(label < 0)[0]
            sim_diag[ood_indices] = self.margin * 2

            ct_diag = sim_diag.expand_as(sim)
            tc_diag = sim_diag.t().expand_as(sim)
            ct_loss = (self.margin + sim - ct_diag).clamp(min=0)
            tc_loss = (self.margin + sim - tc_diag).clamp(min=0)
            ct_loss = ct_loss.masked_fill(mask, 0)
            tc_loss = tc_loss.masked_fill(mask, 0)
            
            contrastive_loss = (ct_loss.sum() + tc_loss.sum()) / (sim.shape[0] * sim.shape[1])

        if self.mode == "nce":
            code_logits = code_features @ text_features.t()
            text_logits = code_logits.t()
            num_logits = code_logits.shape[0]

            labels = torch.arange(num_logits, dtype=torch.int64, device=code_logits.device)
            
            labels = torch.where(label <= 0, -1, labels)
            code_loss = F.cross_entropy(code_logits/0.07, labels, ignore_index=-1)
            text_loss = F.cross_entropy(text_logits/0.07, labels, ignore_index=-1)
            contrastive_loss = (code_loss + text_loss) / 2
        
        return contrastive_loss

class ClassificationLoss(nn.Module):
    def __init__(self, task='codesearch'):
        super(ClassificationLoss, self).__init__()
        self.task = task
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, mm_features, output, label=None):
        labels = torch.where(label < 0, 0, 1).to(output.device)
        classification_loss = F.binary_cross_entropy_with_logits(output, labels.float())

        if isinstance(mm_features, tuple):
            code_info, text_info = mm_features
            classification_loss = classification_loss + torch.mean(code_info) + torch.mean(text_info)
            
            
        return classification_loss