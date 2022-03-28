import json
import os
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
from load_data import *

import argparse
import wandb

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
        # print("labels: ", labels)
        targets_c = labels.take([c], axis=1).ravel()
        # print("targets_c: ", targets_c)
        preds_c = probs.take([c], axis=1).ravel()
        # print("preds_c: ", preds_c)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        # print("precision, recall: ", precision, recall)
        score[c] = sklearn.metrics.auc(recall, precision)
        # print("score: ", score)
    return np.average(score) * 100.0


def compute_metrics(pred, labels):
    """ validation을 위한 metrics function """
    # labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    # probs = pred.predictions
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
    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens={
        'origin':[],
        'entity':["[ENT]", "[/ENT]"],
        'type_entity':["[ORG]","[/ORG]","[PER]","[/PER]","[POH]","[/POH]","[LOC]","[/LOC]","[DAT]","[/DAT]","[NOH]","[/NOH]"],
        'sub_obj':['[SUB_ENT]','[/SUB_ENT]','[OBJ_ENT]','[/OBJ_ENT]']
    }
    num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":special_tokens.get(args.token_type, [])})    

    # load dataset
    dataset = load_data("../dataset/train/train.csv", token_type=args.token_type)

    train_dataset, valid_dataset = train_test_split(dataset, test_size=args.val_ratio, shuffle=True, stratify=dataset['label'], random_state=args.seed)

    train_label = label_to_num(train_dataset['label'].values)
    valid_label = label_to_num(valid_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.resize_token_embeddings(num_added_token + tokenizer.vocab_size)
    print(model.config)
    model.to(device)
    
    train_loader = DataLoader(RE_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    valid_loader = DataLoader(RE_valid_dataset, batch_size=args.valid_batch_size, shuffle=True, drop_last = False)

    optim = AdamW(model.parameters(), lr=args.lr)

    save_path = args.save_dir

    model_config_parameters = {
        "model":args.model,
        "seed":args.seed,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "token_type":args.token_type,
        "optimizer":args.optimizer,
        "lr":args.lr,
        "val_ratio":args.val_ratio,
    }

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "model_config_parameters.json"), 'w') as f:
        json.dump(model_config_parameters, f, indent=4)
    print(f"{save_path}에 model_config_parameter.json 파일 저장")

    tokenizer.save_pretrained(save_path)
    print(f"{save_path}에 tokenizer 저장")

    if not args.wandb == "False":
        wandb.init(project=args.project_name, entity="salt-bread", name=args.report_name, config=model_config_parameters)

    best_eval_loss = 1e9
    best_eval_f1 = 0
    total_idx = 0

    for epoch in range(args.epochs):
        total_f1, total_loss, total_acc = 0, 0, 0
        average_loss, average_f1, average_acc = 0,0,0

        model.train()
        
        for idx, batch in enumerate(tqdm(train_loader)):
            total_idx += 1

            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids =  batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,token_type_ids=token_type_ids)
            pred = outputs[1]
            metric = compute_metrics(pred, labels)

            # loss = outputs[0]
            criterion = create_criterion(args.criterion)

            loss = criterion(pred, labels)

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
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        token_type_ids =  batch['token_type_ids'].to(device)
                        labels = batch['labels'].to(device)
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
                        pred = outputs[1]
                        eval_metric = compute_metrics(pred, labels)

                        # loss = outputs[0]
                        loss = criterion(pred, labels)

                        eval_total_loss += loss
                        eval_total_f1 += eval_metric['micro f1 score']
                        eval_total_auprc += eval_metric['auprc']
                        eval_total_acc += eval_metric['accuracy']

                    eval_average_loss = eval_total_loss/len(valid_loader)
                    eval_average_f1 = eval_total_f1/len(valid_loader)
                    eval_total_auprc = eval_total_auprc/len(valid_loader)
                    eval_average_acc = eval_total_acc/len(valid_loader)

                    if args.checkpoint:
                        model.save_pretrained(os.path.join(save_path, f"checkpoint-{total_idx}"))

                    if eval_average_loss < best_eval_loss:
                        model.save_pretrained(os.path.join(save_path, "best_loss"))
                        best_eval_loss = eval_average_loss

                    if eval_average_f1 > best_eval_f1:
                        model.save_pretrained(os.path.join(save_path, "best_f1"))
                        best_eval_f1 = eval_average_f1

                    if not args.wandb == "False":
                        wandb.log({
                            "epoch":epoch+1,
                            "eval_loss":eval_average_loss,
                            "eval_f1":eval_average_f1,
                            "eval_acc":eval_average_acc
                            })

                    print(f"[EVAL][loss:{eval_average_loss:4.2f} | auprc:{eval_total_auprc/eval_total_idx:4.2f} | ", end="")
                    print(f"micro_f1_score:{eval_average_f1:4.2f} | accuracy:{eval_average_acc:4.2f}]")

                print("--------------------------------------------------------------------------")

            if not args.wandb == "False":
                wandb.log({
                    "epoch":epoch+1,
                    "train_loss":average_loss,
                    "train_f1":average_f1,
                    "train_acc":average_acc
                    })
    
    if not args.wandb == "False":
        wandb.finish()
    
def main(args):
    seed_everything(args.seed)
    train(args)

if __name__ == '__main__':
    # wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="klue/bert-base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--logging_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default="cross_entropy")
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--report_name', type=str)
    parser.add_argument('--project_name', type=str, default="salt_v2")
    parser.add_argument('--token_type', type=str, default="origin") # origin, entity, type_entity, sub_obj, special_entity
    parser.add_argument('--wandb', type=bool, default=True)

    args = parser.parse_args()
    main(args)
