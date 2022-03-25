import os
import torch
import re
import pandas as pd
import pickle as pickle


class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        # print(item)
        return item

    def __len__(self):
        return len(self.labels)
    
class Preprocessing_dataset:
    def __init__(self, dataset, token_type="origin"):
        self.dataset=dataset
        self.token_type=token_type
        
    def return_dataset(self):        
        if self.token_type=='entity':
            return self.preprocessing_dataset_v1(self.dataset)
        elif self.token_type=='type_entity':
            return self.preprocessing_dataset_v2(self.dataset)
        elif self.token_type=='sub_obj':
            return self.preprocessing_dataset_v3(self.dataset)
        else:
            return self.preprocessing_dataset(self.dataset)
        
    def preprocessing_dataset_v1(self,dataset):
        sentences = []
        subject_entity = []
        object_entity = []
        entity_start_token, entity_end_token = "[ENT]", "[/ENT]"
        # word_indices = []

        for sentence, subject, object in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
            subject_word = subject.split(",")[0].split(":")[1].strip()
            object_word = object.split(",")[0].split(":")[1].strip()

            indices = []

            for i in range(2):
                indices.append(int(subject.split(",")[i - 3].split(":")[1]))
                indices.append(int(object.split(",")[i - 3].split(":")[1]))

            indices.sort(reverse=True)

            for idx, entity_token in zip(indices, [entity_end_token, entity_start_token] * 2):
                if entity_token == entity_end_token:
                    sentence = sentence[:idx + 1] + entity_token + sentence[idx + 1:]
                else:
                    sentence = sentence[:idx] + entity_token + sentence[idx:]

            subject_entity.append(subject_word)
            object_entity.append(object_word)
            sentences.append(sentence)
            # word_indices.append(sorted(indices, reverse=True))


        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def preprocessing_dataset(self,dataset):
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1][2:-1]
            j = j[1:-1].split(',')[0].split(':')[1][2:-1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def String2dict(self,string):
        string=re.sub("['{}]","",string)

        entity_dict=dict()
        string=re.sub(',\s(?=start_idx|end_idx|type)',"|",string)

        for pair in string.split('|'):
            # print(pair)
            key,value=pair.split(': ')[0],": ".join(pair.split(': ')[1:])
            entity_dict[key]=value

        return entity_dict

    def preprocessing_dataset_v2(self,dataset):
        """entity를 위한 스페셜 토큰 추가해주는 전처리 함수"""
        subject_entity = []
        object_entity = []
        sentences = []
        for sentence, sub, obj in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
            sub=self.String2dict(sub)
            obj=self.String2dict(obj)

            sentence=sentence[:int(sub['start_idx'])]+"S"*len(sub['word'])+sentence[int(sub['end_idx'])+1:]
            sub_token="["+sub['type']+"]"+sub['word']+"[/"+sub['type']+"]"

            sentence=sentence[:int(obj['start_idx'])]+"O"*len(obj['word'])+sentence[int(obj['end_idx'])+1:]
            obj_token="["+obj['type']+"]"+obj['word']+"[/"+obj['type']+"]"

            sentence=sentence.replace("S"*len(sub['word']),sub_token)
            sentence=sentence.replace("O"*len(obj['word']),obj_token)

            sentences.append(sentence)
            subject_entity.append(sub['word'])
            object_entity.append(obj['word'])
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset
    
    def preprocessing_dataset_v3(self, dataset):
        pre_sentence = []
        subject_entity = []
        object_entity = []
        for s,i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1][2:-1]
            j = j[1:-1].split(',')[0].split(':')[1][2:-1]
            s = re.sub(i, '[SUB_ENT]'+i+'[/SUB_ENT]', s)
            s = re.sub(j, '[OBJ_ENT]'+j+'[/OBJ_ENT]', s)

            subject_entity.append(i)
            object_entity.append(j)
            pre_sentence.append(s)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':pre_sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

def load_data(dataset_dir, token_type='origin'):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = Preprocessing_dataset(pd_dataset,token_type).return_dataset()
    print("preprocessing finished")

    return dataset

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    
    return tokenized_sentences
