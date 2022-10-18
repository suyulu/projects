import torch
import torch.nn as nn
import torch.nn.functional as F


'''
    LSTM model
    
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

'''


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout, device):
        super().__init__()
        self.device = device
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        # out.size() --> batch_size, sequence_length, hidden_size 
        # out[:, -1, :] --> batch_size, sequence_length, hidden_size --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim
        out = torch.sigmoid(out)
        return out


class FocalLoss(torch.nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction
  
  def forward(self, pred, target):
    alpha = self.alpha
    loss = - alpha * (1 - pred) ** self.gamma * target * torch.log(pred) - \
        (1 - alpha) * pred ** self.gamma * (1 - target) * torch.log(1 - pred)
    if self.reduction == 'elementwise_mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    return loss





