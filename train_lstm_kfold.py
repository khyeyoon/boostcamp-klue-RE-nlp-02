import json
from lib2to3.pgen2 import token
import os
from tabnanny import verbose
import torch
import random
import warnings
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loss import create_criterion
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from load_data_lstm import *
from transformers import AutoConfig, AutoModel
from torch.utils.data import IterableDataset
from collate_fn import collate_fn
from functools import partial
import argparse
import wandb
from batch_sampler import bucketed_batch_indices
from model import *
from sklearn.model_selection import StratifiedKFold
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)

    return np.average(score) * 100.0


def compute_metrics(pred, labels):
    """ validation을 위한 metrics function """

    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = pred.argmax(-1)
    probs = pred

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    
    return num_label

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    # load model and tokenizer
    data_idx = random.randint(0, 100)

    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens={
        'origin':[],
        'special_type_entity':[],
        'entity':["[ENT]", "[/ENT]"],
        'type_entity':["[ORG]","[/ORG]","[PER]","[/PER]","[POH]","[/POH]","[LOC]","[/LOC]","[DAT]","[/DAT]","[NOH]","[/NOH]"],
        'type_entity_v2':["[S:ORG]","[/S:ORG]","[S:PER]","[/S:PER]","[S:POH]","[/S:POH]","[S:LOC]","[/S:LOC]","[S:DAT]","[/S:DAT]","[S:NOH]","[/S:NOH]",\
            "[O:ORG]","[/O:ORG]","[O:PER]","[/O:PER]","[O:POH]","[/O:POH]","[O:LOC]","[/O:LOC]","[O:DAT]","[/O:DAT]","[O:NOH]","[/O:NOH]"],
        'sub_obj':['[SUB_ENT]','[/SUB_ENT]','[OBJ_ENT]','[/OBJ_ENT]'],
        'type': ["[S:ORG]","[S:PER]","[S:POH]","[S:LOC]","[S:DAT]","[S:NOH]",\
            "[O:ORG]","[O:PER]","[O:POH]","[O:LOC]","[O:DAT]","[O:NOH]"],
    }
    num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(args.token_type, [])})    

    skf = StratifiedKFold(n_splits=args.kfold_splits, random_state = args.seed, shuffle=True)

    # load dataset
    dataset = load_data("../dataset/train/train.csv", token_type=args.token_type)
    dataset_label = label_to_num(dataset['labels'].values)

    for kfold_idx, (train_idx, valid_idx) in enumerate(skf.split(dataset, dataset_label)):
        print(f'########  kfold : {kfold_idx} start  ########')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30
        model_config.hidden_dropout_prob = args.dropout
        model_config.attention_probs_dropout_prob = args.dropout
        model = AutoModel.from_pretrained(MODEL_NAME, config=model_config)
        model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)
        print(model_config)
        
        if MODEL_NAME=='klue/bert-base':
            h=768
        elif MODEL_NAME=='klue/roberta-large':
            h=1024

        model=RE_LSTM(backbone=model, tokenizer=tokenizer, class_num=model_config.num_labels, hidden_size=h, drop_rate=args.dropout, lstm_layer=args.lstm_layer)
        # model.load_state_dict(torch.load('/opt/ml/baseline/BERT_BiLSTM17/best_loss'+'/pytorch_model.bin'))
        model.to(device)

        train_dataset, valid_dataset = dataset.iloc[train_idx,:], dataset.iloc[valid_idx,:]

        train_label = label_to_num(train_dataset['labels'].values)
        valid_label = label_to_num(valid_dataset['labels'].values)

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer,sep_type=args.sep_type)
        tokenized_valid = tokenized_dataset(valid_dataset, tokenizer,sep_type=args.sep_type)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
        print("[dataset 예시]", tokenizer.decode(RE_train_dataset[data_idx]['input_ids']), sep='\n')

        train_loader = DataLoader(RE_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True,collate_fn=partial(collate_fn, sep=tokenizer.sep_token_id))
        valid_loader = DataLoader(RE_valid_dataset, batch_size=args.valid_batch_size, shuffle=False, drop_last = False,collate_fn=partial(collate_fn, sep=tokenizer.sep_token_id))

        optim = AdamW([{'params': model.Backbone.parameters()}, 
                        {'params':model.fc.parameters()},
                        {'params':model.label_classifier.parameters()},
                        {'params': model.LSTM.parameters(), 'lr': 0.001},], lr=args.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=1, factor=0.5, verbose=True)
        criterion = create_criterion(args.criterion)

        save_path = args.save_dir
        k_save_path = os.path.join(save_path, str(kfold_idx))
        os.makedirs(k_save_path, exist_ok=True)

        model_config_parameters = {
            "model":args.model,
            "seed":args.seed,
            "epochs":args.epochs,
            "batch_size":args.batch_size,
            "token_type":args.token_type,
            "optimizer":args.optimizer,
            "lr":args.lr,
            "val_ratio":args.val_ratio,
            "sep_type":args.sep_type
        }

        if not os.path.exists(k_save_path):
            os.mkdir(k_save_path)

        os.makedirs(k_save_path, exist_ok=True)
        with open(os.path.join(k_save_path, "model_config_parameters.json"), 'w') as f:
            json.dump(model_config_parameters, f, indent=4)
        print(f"{k_save_path}에 model_config_parameter.json 파일 저장")

        tokenizer.save_pretrained(k_save_path)
        print(f"{k_save_path}에 tokenizer 저장")

        if args.wandb == "True":
            wandb.init(project=args.project_name, entity="salt-bread", name=args.report_name+str(kfold_idx), config=model_config_parameters)

        best_eval_loss = 1e9
        best_eval_f1 = 0
        total_idx = 0

        for epoch in range(args.epochs):
            total_f1, total_loss, total_acc = 0, 0, 0
            average_loss, average_f1, average_acc = 0,0,0

            for idx, batch in enumerate(tqdm(train_loader)):
                model.train()
                total_idx += 1
                # print(batch['input_ids'].shape)

                optim.zero_grad()
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                batch['token_type_ids'] =  batch['token_type_ids'].to(device)
                batch['labels'] = batch['labels'].to(device)
                # labels = batch['labels'].to(device)
                # batch=batch.to(device)
                preds = model(batch)
                # print(preds)
                
                # pred = outputs[1]
                # preds = torch.argmax(outputs, dim=-1)
                metric = compute_metrics(preds, batch['labels'])

                # loss = outputs[0]
                loss = criterion(preds, batch['labels'])

                loss.backward()
                optim.step()
                total_loss += loss
                total_f1 += metric['micro f1 score']
                # total_auprc += metric['auprc']
                total_acc += metric['accuracy']

                average_loss = total_loss/(idx+1)
                average_f1 = total_f1/(idx+1)
                average_acc = total_acc/(idx+1)

                if idx%args.logging_step == 0:
                    print(f"[TRAIN][EPOCH:({epoch + 1}/{args.epochs}) | loss:{average_loss:4.2f} | ", end="")
                    print(f"micro_f1_score:{average_f1:4.2f} | accuracy:{average_acc:4.2f}]")

            
                if total_idx%args.eval_step == 0:
                    eval_total_loss, eval_total_f1, eval_total_auprc, eval_total_acc = 0, 0, 0, 0
                    with torch.no_grad():
                        model.eval()

                        print("--------------------------------------------------------------------------")
                        print(f"[EVAL] STEP:{total_idx}, BATCH SIZE:{args.batch_size}")
                        for idx, batch in enumerate(tqdm(valid_loader)):

                            batch['input_ids'] = batch['input_ids'].to(device)
                            batch['attention_mask'] = batch['attention_mask'].to(device)
                            batch['token_type_ids'] =  batch['token_type_ids'].to(device)
                            labels = batch['labels'].to(device)
                            preds = model(batch)
                            eval_metric = compute_metrics(preds, labels)

                            # loss = outputs[0]
                            loss = criterion(preds, labels)

                            eval_total_loss += loss
                            eval_total_f1 += eval_metric['micro f1 score']
                            eval_total_auprc += eval_metric['auprc']
                            eval_total_acc += eval_metric['accuracy']

                        eval_average_loss = eval_total_loss/len(valid_loader)
                        eval_average_f1 = eval_total_f1/len(valid_loader)
                        eval_total_auprc = eval_total_auprc/len(valid_loader)
                        eval_average_acc = eval_total_acc/len(valid_loader)

                        if args.checkpoint:
                            model.save_pretrained(os.path.join(k_save_path, f"checkpoint-{total_idx}"))

                        if eval_average_loss < best_eval_loss:
                            PATH=os.path.join(k_save_path, "best_loss")
                            os.makedirs(PATH, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(PATH, "pytorch_model.bin"))
                            # model.save_pretrained(os.path.join(save_path, "best_loss"))
                            best_eval_loss = eval_average_loss

                        if eval_average_f1 > best_eval_f1:
                            PATH=os.path.join(k_save_path, "best_f1")
                            os.makedirs(PATH, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(PATH, "pytorch_model.bin"))
                            # model.save_pretrained(os.path.join(save_path, "best_f1"))
                            best_eval_f1 = eval_average_f1

                        if args.wandb == "True":
                            wandb.log({
                                "step":total_idx,
                                "eval_loss":eval_average_loss,
                                "eval_f1":eval_average_f1,
                                "eval_acc":eval_average_acc
                                })

                        lr_scheduler.step(eval_average_loss)
                        # 현재 lr        
                        for param_group in optim.param_groups:
                            print(param_group['lr'])

                        print(f"[EVAL][loss:{eval_average_loss:4.2f} | auprc:{eval_total_auprc:4.2f} | ", end="")
                        print(f"micro_f1_score:{eval_average_f1:4.2f} | accuracy:{eval_average_acc:4.2f}]")

                    print("--------------------------------------------------------------------------")


            if args.wandb == "True":
                wandb.log({
                    "epoch":epoch+1,
                    "train_loss":average_loss,
                    "train_f1":average_f1,
                    "train_acc":average_acc
                    })
        
        if args.wandb == "True":
            wandb.finish()
        time.sleep(30)
    
def main(args):
    seed_everything(args.seed)
    train(args)

if __name__ == '__main__':
    # wandb.login()
    # python train_lstm_kfold.py --wandb False --model klue/roberta-large --epochs 5 --eval_step 40 --criterion focal --token_type type_entity --sep_type ENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="klue/bert-base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--logging_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default="cross_entropy") # 'cross_entropy', 'focal', 'label_smoothing', 'f1'
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--report_name', type=str)
    parser.add_argument('--project_name', type=str, default="salt_v3")
    parser.add_argument('--token_type', type=str, default="origin") # origin, entity, type_entity, sub_obj, special_entity
    parser.add_argument('--wandb', type=str, default="True")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--sep_type', type=str, default='SEP')
    parser.add_argument('--lstm_layer', type=int, default=1)
    parser.add_argument('--kfold_splits', type=int, default=5)

    args = parser.parse_args()
    main(args)