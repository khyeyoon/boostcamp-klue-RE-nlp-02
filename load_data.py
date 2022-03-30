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
        self.dict_type_to_str = {
            "PER":"person",
            "ORG":"organization",
            "POH":"position",
            "LOC":"location",
            "DAT":"date",
            "NOH":"number"
        }
        
    def return_dataset(self):        
        if self.token_type=='entity':
            return self.preprocessing_dataset_entity(self.dataset)
        elif self.token_type=='type_entity':
            return self.preprocessing_dataset_type_entity(self.dataset)
        elif self.token_type=='sub_obj':
            return self.preprocessing_dataset_sub_obj(self.dataset)
        elif self.token_type=='special_entity':
            return self.preprocessing_dataset_special_entity(self.dataset)
        elif self.token_type=='special_type_entity':
            return self.preprocessing_dataset_special_type_entity(self.dataset)
        else:
            if not self.token_type=='origin':
                print(f"{self.token_type}에 해당하는 token_type이 없어 origin으로 Preprocessing 합니다.")
            return self.preprocessing_dataset(self.dataset)
        
    def preprocessing_dataset_entity(self,dataset):
        sentences = []
        subject_entity = []
        object_entity = []
        entity_start_token, entity_end_token = "[ENT]", "[/ENT]"

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

        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
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

    def preprocessing_dataset_type_entity(self,dataset):
        """entity를 위한 스페셜 토큰 추가해주는 전처리 함수"""
        subject_entity = []
        object_entity = []
        sub_type_entity=[]
        obj_type_entity=[]
        sentences = []
        for sentence, sub, obj, in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
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
            sub_type_entity.append(sub['type'])
            obj_type_entity.append(obj['type'])

        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'sub_type_entity':sub_type_entity,'obj_type_entity':obj_type_entity, 'label': dataset['label']})
        
        return out_dataset
    
    def preprocessing_dataset_sub_obj(self, dataset):
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

    def preprocessing_dataset_special_entity(self, dataset):
        pre_sentence = []
        subject_entity = []
        object_entity = []
        for s,i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1][2:-1]
            j = j[1:-1].split(',')[0].split(':')[1][2:-1]
            s = re.sub(i, ' @ '+i+' @ ', s)
            s = re.sub(j, ' # '+j+' # ', s)

            subject_entity.append(i)
            object_entity.append(j)
            pre_sentence.append(s)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':pre_sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def preprocessing_dataset_special_type_entity(self, dataset):
        pre_sentence = []
        subject_entity = []
        object_entity = []
        for s, sub, obj in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
            sub_entity = sub[1:-1].split(',')[0].split(':')[1][2:-1]
            obj_entity = obj[1:-1].split(',')[0].split(':')[1][2:-1]
            sub_type = self.dict_type_to_str[sub[1:-1].split(',')[-1].split(':')[1].strip().replace("'", "")]
            obj_type = self.dict_type_to_str[obj[1:-1].split(',')[-1].split(':')[1].strip().replace("'", "")]

            s = re.sub(sub, ' @ * ' + sub_type + ' * ' + sub_entity + ' @ ', s)
            s = re.sub(obj, ' # ^ ' + obj_type + ' ^ ' + obj_entity + ' # ', s)

            subject_entity.append(sub)
            object_entity.append(obj)
            pre_sentence.append(s)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':pre_sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset


    def preprocessing_dataset(self, dataset):
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1][2:-1]
            j = j[1:-1].split(',')[0].split(':')[1][2:-1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

def load_data(dataset_dir, token_type='origin', is_relation=False):
    """
    Arguments:
        dataset_dir (dataset path): dataset 파일 경로
        token_type (str, optional): entity token 추가 방법
            Should be one of
            - 'origin' : entity token 추가하지 않음
            - 'entity' : [ENT], [/ENT] token 추가
            - 'type_entity' : word type으로 token 추가
            - 'sub_obj' : subject와 object 각각 token 추가
            - 'special_entity' : @, #으로 token 추가
    """
    if is_relation:
        pd_dataset = pd.read_csv(dataset_dir)
        pd_dataset = pd_dataset[pd_dataset.label != 'no_relation']
    else:
        pd_dataset = pd.read_csv(dataset_dir)

    dataset = Preprocessing_dataset(pd_dataset,token_type).return_dataset()
    print(f"{token_type} preprocessing finished")

    return dataset

def tokenized_dataset(dataset, tokenizer, sep_type='SEP'):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    if sep_type.upper()=='SEP':
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)

    elif sep_type.upper()=='ENT':
        for e01, e02, sub, obj in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sub_type_entity'], dataset['obj_type_entity']):
            temp = ''
            sub_token="["+sub+"]"+e01+"[/"+sub+"]"
            obj_token="["+obj+"]"+e02+"[/"+obj+"]"
            temp = sub_token + obj_token  
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
