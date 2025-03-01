import os
import torch
import random
import sklearn
import numpy as np
import pandas as pd
import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
# from transformers import BertTokenizerFast, GPT2LMHeadModel
from load_data import *

import argparse
import wandb


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


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

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

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # gpt3-kor-small_based_on_gpt2
    # tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
    # input_ids = tokenizer.encode("text to tokenize")[1:]  # remove cls token
            
    # MODEL_NAME = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")

    # load dataset
    dataset = load_data("../dataset/train/train.csv")
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
        
    train_dataset, valid_dataset = train_test_split(dataset, test_size=args.val_ratio, shuffle=True, stratify=dataset['label'], random_state=args.seed)

    train_label = label_to_num(train_dataset['label'].values)
    valid_label = label_to_num(valid_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =    AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =   AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)
    
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.save_dir,           # output directory
        save_total_limit=5,               # number of total save model.
        save_steps=500,                   # model saving step.
        num_train_epochs=args.epochs,      # total number of training epochs
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,    # batch size for evaluation
        warmup_steps=500,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                # strength of weight decay
        logging_dir='./logs',             # directory for storing logs
        logging_steps=100,                # log saving step.
        evaluation_strategy='steps',      # evaluation strategy to adopt during training
                                          # `no`: No evaluation during training.
                                          # `steps`: Evaluate every `eval_steps`.
                                          # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,                 # evaluation step.
        load_best_model_at_end = True, 
        report_to="wandb",  # enable logging to W&B
    )
    trainer = Trainer(
        model=model,                      # the instantiated 🤗 Transformers model to be trained
        args=training_args,               # training arguments, defined above
        train_dataset=RE_train_dataset,   # training dataset
        eval_dataset=RE_valid_dataset,    # evaluation dataset
        compute_metrics=compute_metrics   # define metrics function
    )

    # train model
    trainer.train()
    best_save_path = args.best_save_dir
    model.save_pretrained(best_save_path)
    tokenizer.save_pretrained(best_save_path)
    wandb.finish()
    
def main(args):
    seed_everything(args.seed)
    wandb.init(project=args.project_name, entity="salt-bread", name=args.report_name)
    train(args)

if __name__ == '__main__':
    # wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="klue/bert-base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--best_save_dir', type=str, default="./best_model")
    parser.add_argument('--report_name', type=str)
    parser.add_argument('--project_name', type=str, default="salt_v1")

    args = parser.parse_args()
    main(args)