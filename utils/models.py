import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel, ElectraTokenizer, ElectraForSequenceClassification, ElectraModel




def load_tokenizer_and_model(model_args, device):
    # load tokenizer
    special_tokens={
        'origin':[],
        'entity':["[ENT]", "[/ENT]"],
        'type_entity':["[ORG]","[/ORG]","[PER]","[/PER]","[POH]","[/POH]","[LOC]","[/LOC]","[DAT]","[/DAT]","[NOH]","[/NOH]"],
        'sub_obj':['[SUB_ENT]','[/SUB_ENT]','[OBJ_ENT]','[/OBJ_ENT]']
    }

    MODEL_NAME = model_args['model_name']
    dropout = model_args['dropout']
    model_case = model_args['model_case']
    token_type = model_args['token_type']
    class_num = model_args['class_num']

    special_tokens[token_type] += model_args['append_tokens']
    #print(special_tokens[token_type])

    # model config
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = class_num
    model_config.hidden_dropout_prob = dropout
    model_config.attention_probs_dropout_prob = dropout


    if model_args['use_lstm']:
        if model_case == 'automodel':
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(token_type, [])})    

            b_model = AutoModel.from_pretrained(MODEL_NAME, config= model_config)
            b_model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)

        elif model_case == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
            num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(token_type, [])})    
        
            b_model = ElectraModel.from_pretrained(MODEL_NAME, config= model_config)
            b_model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)
        
        b_model_output_dim = b_model.pooler.dense.out_features
        model = TunedModelLSTM(b_model, b_model_output_dim, class_num, device, dropout, lstm_layers = model_args['lstm_layers'])

    else:
        if model_case == 'automodel':
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(token_type, [])})    

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)
            print(model.config)


        elif model_case == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
            num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(token_type, [])})    
            
            model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config= model_config)
            model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)
            print(model.config)


    print(f'{model_case} {MODEL_NAME} model and tokenizer loaded')
    return tokenizer, model




class TunedModelLSTM(nn.Module):
    def __init__(self, base_model, b_model_output_dim, num_classes, device, dropout_ratio, lstm_layers , lstm_dim = 512, bidirectional = True):
        super().__init__()
        self.base_model= base_model
        self.num_classes = num_classes
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.device = device
        self.b_model_output_dim=b_model_output_dim
        self.bidirectional=bidirectional


        self.linear = nn.Linear(self.b_model_output_dim, self.num_classes)
        self.dropout_cls = nn.Dropout(dropout_ratio)

        self.rnn = LSTM(self.device, self.b_model_output_dim, self.lstm_dim, self.lstm_layers, self.bidirectional)

        # self.rnn = nn.LSTM(
        #     input_size = self.b_model_output_dim, 
        #     hidden_size = self.lstm_dim, 
        #     num_layers = self.lstm_layers, 
        #     batch_first = True)
        self.rnn_lin = nn.Linear(self.lstm_dim*2 if self.bidirectional else self.lstm_dim, self. num_classes)
        self.dropout_lhs = nn.Dropout(dropout_ratio)

        #self.loss = loss
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids)

        lhs = outputs.last_hidden_state
        # h0 = torch.zeros(self.lstm_layers, lhs.size(0), self.lstm_dim).to(self.device) # 2, batch, dim
        # c0 = torch.zeros(self.lstm_layers, lhs.size(0), self.lstm_dim).to(self.device) 
        # rnn_out, (hn, cn) = self.rnn(lhs, (h0, c0))
        rnn_out, (hn, cn) = self.rnn(lhs)
        lhs_logits = self.rnn_lin(rnn_out[:,-1:]).view([-1, self.num_classes])
        lhs_logits = self.dropout_cls(lhs_logits)
        
        cls = outputs.pooler_output
        logits = self.dropout_cls(self.linear(cls))
        
        logits_cat = torch.cat((logits.unsqueeze(dim=0), lhs_logits.unsqueeze(dim=0)), dim = 0)
        pred = torch.mean(logits_cat, dim=0)
        
        if isinstance(labels, torch.Tensor):
            return [None, pred]
        else:
            return [pred]


class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers = 2, bidirectional = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.h_c_layers = num_layers*2 if bidirectional else num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.h_c_layers, x.shape[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.h_c_layers, x.shape[0], self.hidden_size).to(self.device)
        out, (h_, c_) = self.lstm(x, (h0,c0))
        return out, (h_,c_)