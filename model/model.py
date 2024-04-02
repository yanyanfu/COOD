from typing import Any, Union, List, Dict, Tuple
from tqdm import tqdm
import numpy as np

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig, RobertaModel



class LinearLayer(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], dropout=None, bias=True, act=nn.Tanh):
        super(LinearLayer, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class AnomalyDetectionModel(nn.Module):
    def __init__(self,
                 pretrained: bool = True,
                 encoder: str = "roberta",
                 task: str = "codesearch",
                 num_features: int = 768,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 classifier: str = "sparse",
                 freeze_layers: Dict[str, List] = None):

        super(AnomalyDetectionModel, self).__init__()

        # pretrained model
        config = RobertaConfig.from_pretrained(encoder)
        config.num_labels = num_classes
                                              
        print('Loading pretrained model:', encoder)
        self.encoder = RobertaModel.from_pretrained(encoder,
                                                    from_tf=bool('.ckpt' in encoder),
                                                    config=config)

        # freeze layers
        if freeze_layers is not None:
            print(f"Freezing transformer layers: {freeze_layers}")
            for layer in freeze_layers.keys():
                for name, param in self.clip_model.named_parameters():
                    if layer in name:
                        if len(freeze_layers[layer]) > 0:
                            for l in freeze_layers[layer]:
                                if str(l) in name.split("."):
                                    param.requires_grad = False
                        else:
                            param.requires_grad = False

        self.task = task
        self.num_classes = num_classes
        self.dropout = dropout
        self.classifier = classifier
        self.code_info = LinearLayer((num_features, num_features))
        self.text_info = LinearLayer((num_features, num_features))
        self.anom_class = LinearLayer((num_features * 4, num_features // 2, 1), dropout=dropout, act=nn.Tanh)

    def forward(self, code, text):
        code_features = self.encoder(code, attention_mask=code.ne(1))[0]
        code_features = (code_features*code.ne(1)[:,:,None]).sum(1)/code.ne(1).sum(-1)[:,None]
        text_features = self.encoder(text, attention_mask=text.ne(1))[0]
        text_features = (text_features*text.ne(1)[:,:,None]).sum(1)/text.ne(1).sum(-1)[:,None]

        # feature concatenation
        mm_features = torch.cat([code_features, text_features, text_features - code_features, text_features + code_features], dim=1)

        # anomaly detection
        anom_output = self.anom_class(mm_features).squeeze(-1)

        # normalize features
        code_features = code_features / code_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return (code_features, text_features), mm_features, anom_output
