import torch
from torch import nn
from dataset_utils import *
from ensemble_utils import *

class StackedEnsembleFC(nn.Module):
    def __init__(self, n_models=3, dropout_rate=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768*4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
        )
        self.ensemble_weight = nn.Parameter(torch.empty(1, 1, 1, n_models), requires_grad=True)
        nn.init.xavier_uniform_(self.ensemble_weight)

    def forward(self, hidden_layers):
        x = torch.sum(hidden_layers*torch.softmax(self.ensemble_weight, dim=-1), dim=-1)
        logits = self.fc(x)
        return logits
    
class StackedEnsembleBiLSTM(nn.Module):
    def __init__(self, n_models=3, dropout_rate=0.2):
        super().__init__()
        self.LSTM = nn.LSTM(768*4, 512, bidirectional=True, bias=True, batch_first=True)
        self.ReLU = nn.ReLU()
        self.Linear = nn.Linear(512*2, 15)
        self.ensemble_weight = nn.Parameter(torch.empty(1, 1, 1, n_models), requires_grad=True)
        nn.init.xavier_uniform_(self.ensemble_weight)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_layers):
        x = torch.sum(hidden_layers*torch.softmax(self.ensemble_weight, dim=-1), dim=-1)
        x = self.dropout(x)
        x, _ = self.LSTM(x)
        logits = self.Linear(self.ReLU(x))
        return logits