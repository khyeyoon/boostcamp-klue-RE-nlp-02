import torch
import torch.nn as nn
import numpy as np

class TunedModelLinear(nn.Module):
    def __init__(self, base_model, num_classes, seq_len = 256, loss = nn.CrossEntropyLoss()):
        super().__init__()
        self.base_model= base_model
        self.linear = nn.Linear(768, num_classes)
        self.lhs_linear = nn.Linear(seq_len, num_classes)
        self.loss = loss
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids)
        lhs = outputs.last_hidden_state
        print('lhs : ', lhs.shape)
        lhs = torch.mean(lhs, dim=-1)
        lhs_logits = self.lhs_linear(lhs)
        
        cls = outputs.pooler_output
        logits = self.linear(cls)
        
        logits_cat = torch.cat((logits.unsqueeze(dim=0), lhs_logits.unsqueeze(dim=0)), dim = 0)
        pred = torch.mean(logits_cat, dim=0)
        
        # if labels:
        #     return [cal_loss(pred, labels), pred]
        return [None,pred]
    
    # def cal_loss(pred, target):
    #     return self.loss(pred, target)
        


class TunedModelLSTM(nn.Module):
    def __init__(self, base_model, num_classes, device, dropout_ratio, n_layer = 1, lstm_dim = 768):
        super().__init__()
        self.base_model= base_model
        self.num_classes = num_classes
        self.lstm_dim = lstm_dim
        self.n_layer = n_layer
        self.device = device

        self.linear = nn.Linear(768, self. num_classes)
        self.dropout_cls = nn.Dropout(dropout_ratio)

        self.rnn = nn.LSTM(
            input_size = 768, 
            hidden_size = self.lstm_dim, 
            num_layers = self.n_layer, 
            dropout = dropout_ratio,
            batch_first = True)
        self.rnn_lin = nn.Linear(self.lstm_dim, self. num_classes)
        self.dropout_lhs = nn.Dropout(dropout_ratio)

        #self.loss = loss
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids)

        lhs = outputs.last_hidden_state
        h0 = torch.zeros(self.n_layer, lhs.size(0), self.lstm_dim).to(self.device) # 1, batch, dim
        c0 = torch.zeros(self.n_layer, lhs.size(0), self.lstm_dim).to(self.device) 
        rnn_out, (hn, cn) = self.rnn(lhs, (h0, c0))
        lhs_logits = self.rnn_lin(rnn_out[:,-1:]).view([-1, self.num_classes])
        lhs_logits = self.dropout_cls(lhs_logits)
        
        cls = outputs.pooler_output
        logits = self.dropout_cls(self.linear(cls))
        
        logits_cat = torch.cat((logits.unsqueeze(dim=0), lhs_logits.unsqueeze(dim=0)), dim = 0)
        pred = torch.mean(logits_cat, dim=0)
        
        if isinstance(labels, torch.Tensor):
        #     return [cal_loss(pred, labels), pred]
            return [None, pred]
        else:
            return [pred]
    
    # def cal_loss(pred, target):
    #     return self.loss(pred, target)