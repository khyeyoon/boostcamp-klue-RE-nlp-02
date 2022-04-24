import torch

class FCLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class RE_NET(torch.nn.Module):
    def __init__(self, backbone, tokenizer, class_num, device, hidden_size=768, drop_rate=0.1):
        super(RE_NET, self).__init__()
        self.tokenizer=tokenizer
        self.Backbone=backbone
        self.device=device
        self.hidden_size=hidden_size
        self.tanh=torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.sub_fc = FCLayer(hidden_size, hidden_size, drop_rate)
        self.obj_fc = FCLayer(hidden_size, hidden_size, drop_rate)
        
        self.cls_fc = FCLayer(hidden_size, hidden_size, drop_rate)
        self.label_classifier = FCLayer(
            hidden_size * 3,
            class_num,
            drop_rate,
            use_activation=False,
        )

    def slicing_feature_ent(self,last_hidden_state,index,length=40):
        # dim 1 사이즈 40으로 맞추기 => output(64,40,768)
        sub_feature_list=[]
        obj_feature_list=[]
        for i in range(last_hidden_state.size(0)):
            s1,e1,s2,e2=index[i]
            # 1,x(길이 모두 다름),768
            # subject entity
            sub_feature_list.append((torch.sum(last_hidden_state[i,s1:e1+1,:]/(e1-s1)),0).unsqueeze(0))
            # object entity
            obj_feature_list.append((torch.sum(last_hidden_state[i,s2:e2+1,:]/(e2-s2)),0).unsqueeze(0))

        # 최종 output : 64,1,768
        return torch.cat(sub_feature_list, dim=0),torch.cat(obj_feature_list, dim=0)


    def forward(self,x):
        # cls input: torch.Size([64, 768]) -> cls_output: torch.Size([64, 30])

        backbone_out = self.Backbone(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])

        pooled_output = backbone_out['pooler_output']
        sub_ent,obj_ent = self.slicing_feature_ent(backbone_out['last_hidden_state'],x['pos'])

        pooled_output=self.cls_fc(pooled_output)
        sub_ent=self.obj_fc(sub_ent)
        obj_ent=self.obj_fc(obj_ent)

        concat_h = torch.cat([pooled_output, sub_ent, obj_ent], dim=-1)
        logits=self.label_classifier(concat_h)

        return logits


class RE_NET_v2(torch.nn.Module):
    """
    [PER]subject[/PER][PER]object[/PER] sentence
    -> sentence 앞에 entity와 sentence 안에 entity feature를 추출해서 예측
    """
    def __init__(self, backbone, tokenizer, class_num, device, hidden_size=768, drop_rate=0.1):
        super(RE_NET_v2, self).__init__()
        self.tokenizer=tokenizer
        self.Backbone=backbone
        self.device=device
        self.hidden_size=hidden_size
        self.tanh=torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=drop_rate)

        self.sub_fc1 = FCLayer(hidden_size, hidden_size, drop_rate)
        self.obj_fc1 = FCLayer(hidden_size, hidden_size, drop_rate)

        self.sub_fc2 = FCLayer(hidden_size, hidden_size, drop_rate)
        self.obj_fc2 = FCLayer(hidden_size, hidden_size, drop_rate)
        
        self.cls_fc = FCLayer(hidden_size, hidden_size, drop_rate)
        self.label_classifier = FCLayer(
            hidden_size * 3,
            class_num,
            drop_rate,
            use_activation=False,
        )

    def slicing_feature_ent(self,last_hidden_state,index,length=40):
        # dim 1 사이즈 40으로 맞추기 => output(64,40,768)
        sub_feature_list1=[]
        obj_feature_list1=[]
        sub_feature_list2=[]
        obj_feature_list2=[]
        for i in range(last_hidden_state.size(0)):
            s1,e1,s2,e2,s3,e3,s4,e4=index[i]
            # 1,x(길이 모두 다름),768
            
            # subject entity
            sub_feature_list1.append(torch.sum((last_hidden_state[i,s1:e1+1,:]/(e1-s1)),0).unsqueeze(0))
            sub_feature_list2.append(torch.sum((last_hidden_state[i,s2:e2+1,:]/(e2-s2)),0).unsqueeze(0))
            # object entity
            obj_feature_list1.append(torch.sum((last_hidden_state[i,s3:e3+1,:]/(e3-s3)),0).unsqueeze(0))
            obj_feature_list2.append(torch.sum((last_hidden_state[i,s4:e4+1,:]/(e4-s4)),0).unsqueeze(0))

        # 최종 output : 64,1,768
        return torch.cat(sub_feature_list1, dim=0),torch.cat(obj_feature_list1, dim=0),torch.cat(sub_feature_list2, dim=0),torch.cat(obj_feature_list2, dim=0)


    def forward(self,x):
        # cls input: torch.Size([64, 768]) -> cls_output: torch.Size([64, 30])
        backbone_out = self.Backbone(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])

        pooled_output = backbone_out['pooler_output']
        sub_ent1,obj_ent1,sub_ent2,obj_ent2 = self.slicing_feature_ent(backbone_out['last_hidden_state'],x['pos'])

        pooled_output=self.cls_fc(pooled_output)

        sub_ent1=self.sub_fc1(sub_ent1)
        obj_ent1=self.obj_fc1(obj_ent1)

        sub_ent2=self.sub_fc2(sub_ent2)
        obj_ent2=self.obj_fc2(obj_ent2)

        concat_h = torch.cat([pooled_output, (sub_ent1+sub_ent2)/2, (obj_ent1+obj_ent2)/2], dim=-1)
        logits=self.label_classifier(concat_h)

        return logits


class LSTM(torch.nn.Module):
    def __init__(self, xdim=768, hdim=768, n_layer=1, drop_rate=None):
        super(LSTM, self).__init__()
        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.xdim = xdim
        self.hdim = hdim

        self.n_layer = n_layer # K

        self.lstm = torch.nn.LSTM(input_size=self.xdim,
                            hidden_size=self.hdim,
                            num_layers=self.n_layer,
                            bidirectional=True,
                            batch_first=True,
                            # dropout=drop_rate,
                            )

        self.fc = FCLayer(self.hdim*2, self.hdim, drop_rate)

    def forward(self, text):
        # text shape : ([64, 244, 768])
        # h0,c0 = [layer*2(bi-direction), batch, hidden]
        h0 = torch.zeros(self.n_layer*2,text.size(0),self.hdim).to(self.device)
        c0 = torch.zeros(self.n_layer*2,text.size(0),self.hdim).to(self.device)

        rnn_out,(hn,cn) = self.lstm(text,(h0,c0))
        # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        rnn_out = rnn_out[:,-1,:]
        out = self.fc(rnn_out)
        # print(out.shape)
        
        return out

class RE_LSTM(torch.nn.Module):
    def __init__(self, backbone, tokenizer, class_num, hidden_size=768, drop_rate=0.1, lstm_layer=2):
        super(RE_LSTM, self).__init__()
        self.tokenizer=tokenizer
        self.Backbone=backbone
        self.fc = FCLayer(hidden_size, hidden_size, drop_rate)
        self.LSTM = LSTM(xdim=hidden_size, hdim=hidden_size, drop_rate=drop_rate,n_layer=lstm_layer)
        self.tanh=torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.label_classifier = FCLayer(
            hidden_size * 2,
            class_num,
            drop_rate,
            use_activation=False,
        )

    def forward(self,x):
        # cls input: torch.Size([64, 768]) -> cls_output: torch.Size([64, 30])
        # lstm input: torch.Size([64, 244, 768]) -> lstm_out: torch.Size([64, 30])

        bert_out = self.Backbone(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])

        lstm_out = self.LSTM(bert_out['last_hidden_state'])
        print(lstm_out.shape)
        cls_out = self.fc(self.tanh(bert_out['pooler_output']))

        # lstm_out=torch.sum(lstm_out,1)
        output=self.label_classifier(torch.cat((cls_out,lstm_out),dim=-1))

        return output