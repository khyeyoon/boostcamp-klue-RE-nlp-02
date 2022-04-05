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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from functools import partial


from utils.loss import *
from utils.models import *
from utils.scheduler import create_lr_scheduler
from utils.load_data import *

import argparse
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    pwd = os.path.dirname( os.path.abspath( __file__ ))
    print(pwd)
    # hold seeds
    seed_everything(args.seed)
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # make save dir
    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)

    # load dataset
    dataset = load_data("../dataset/train/train.csv", token_type=args.token_type)
    dataset_label = label_to_num(dataset['label'].values)

    # models parameter
    model_args = {
        'model_name' : args.model,
        'dropout' : args.dropout,
        'model_case': args.model_case,
        'use_lstm': args.use_lstm,
        'token_type': args.token_type,
        'lstm_layers': args.lstm_layers,
        'class_num' : 30,
        "append_tokens":get_sub_obj(dataset) if args.use_sub_obj else []
    }


    # save model_config_parameters
    model_config_parameters = {
        "model":args.model,
        "seed":args.seed,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "token_type":args.token_type,
        "optimizer":args.optimizer,
        "lr":args.lr,
        "val_ratio":args.val_ratio,
        "sep_type":args.sep_type,
        "kfold_splits":args.kfold_splits,
        "model_case":args.model_case,
        "use_lstm":args.use_lstm,
    }
    with open(os.path.join(save_path, "model_config_parameters.json"), 'w') as f:
        json.dump(model_config_parameters, f, indent=4)
    print(f"{save_path}에 model_config_parameter.json 파일 저장")


    # if kfold
    skf = StratifiedKFold(n_splits=args.kfold_splits, random_state = args.seed, shuffle=True)
    for kfold_idx, (train_idx, valid_idx) in enumerate(skf.split(dataset, dataset_label)):
        print(f'#################  kfold : {kfold_idx} start  #################')
        # load tokenizer, model
        tokenizer, model = load_tokenizer_and_model(model_args, device)

        # save tokenizer
        tokenizer.save_pretrained(save_path)
        print(f"{save_path}에 tokenizer 저장")

        # make report.txt
        k_save_path = os.path.join(save_path, str(kfold_idx))
        os.makedirs(k_save_path, exist_ok=True)
        f = open(os.path.join(k_save_path, "report.txt"), 'w')
        f.close()
        print('make report.txt')

        RE_train_dataset, RE_valid_dataset = make_train_valid_datasets(dataset, train_idx, valid_idx, tokenizer, args.sep_type)

        # get random number to choose example sentence
        data_idx = random.randint(0, 100)
        print("[dataset 예시]", tokenizer.decode(RE_train_dataset[data_idx]['input_ids']), sep='\n')

        model.to(device)

        train_loader = DataLoader(RE_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                drop_last = True, collate_fn=partial(collate_fn, sep=tokenizer.sep_token_id))
        valid_loader = DataLoader(RE_valid_dataset, batch_size=args.valid_batch_size, shuffle=True,
                                 drop_last = False, collate_fn=partial(collate_fn, sep=tokenizer.sep_token_id))

        optim = AdamW(model.parameters(), lr=args.lr)
        # optim = AdamW([
        #                 {'params' : model.base_model.parameters(), 'lr':args.lr},
        #                 {'params':model.linear.parameters(), 'lr':args.lr},
        #                 {'params' : model.rnn.parameters(), 'lr' : 0.001},
        #                 {'params' : model.rnn_lin.parameters(), 'lr' : 0.001},
        #                 ])

        # for param_group in optim.param_groups:
        #     print(param_group, param_group['lr'])

        criterion = create_criterion(args.criterion)

        if args.lr_scheduler:
            scheduler = create_lr_scheduler(args.lr_scheduler, optimizer=optim, mode='min', patience=1, factor=0.5, verbose=True) # 선택에 맞게 변형
            #scheduler = lr_scheduler(optim, 'min', patience=1, factor=0.5, verbose=True) # 선택에 맞게 변형

        if args.wandb == "True":
            name = args.report_name +'_'+ str(kfold_idx)
            wandb.init(project=args.project_name, entity="salt-bread", name=name, config=model_config_parameters)

        best_eval_loss = 1e9
        best_eval_f1 = 0
        total_idx = 0

        for epoch in range(args.epochs):
            total_f1, total_loss, total_acc = 0, 0, 0
            average_loss, average_f1, average_acc = 0,0,0
            
            for idx, batch in enumerate(tqdm(train_loader)):
                model.train()
                total_idx += 1

                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids =  batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
                pred = outputs[1]
                metric = compute_metrics(pred, labels)

                # loss = outputs[0]
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
                    print(f"[K_FOLD:({kfold_idx})][TRAIN][EPOCH:({epoch + 1}/{args.epochs}) | loss:{average_loss:4.2f} | ", end="")
                    print(f"micro_f1_score:{average_f1:4.2f} | accuracy:{average_acc:4.2f}]")

            
                if total_idx%args.eval_step == 0:
                    eval_total_loss, eval_total_f1, eval_total_auprc, eval_total_acc = 0, 0, 0, 0
                    label_list, pred_list = [], []
                    with torch.no_grad():
                        model.eval()
                        print("--------------------------------------------------------------------------")
                        print(f"[K_FOLD:({kfold_idx})][EVAL] STEP:{total_idx}, BATCH SIZE:{args.batch_size}")
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

                            pred_list.extend(list(pred.detach().cpu().numpy().argmax(-1)))
                            label_list.extend(list(labels.detach().cpu().numpy()))

                        eval_average_loss = eval_total_loss/len(valid_loader)
                        eval_average_f1 = eval_total_f1/len(valid_loader)
                        eval_total_auprc = eval_total_auprc/len(valid_loader)
                        eval_average_acc = eval_total_acc/len(valid_loader)

                        cf = confusion_matrix(label_list, pred_list)
                        cf = str_cf(cf)
                        cr = classification_report(label_list, pred_list)
                        with open(os.path.join(k_save_path, "report.txt"), "a") as f:
                            f.write("#"*10 + f"  {total_idx}  " + "#"*100 + '\n')
                            f.write(cf)
                            f.write(cr)
                            f.write('\n')

                        if args.use_lstm:
                            if args.checkpoint:
                                os.makedirs( os.path.join(k_save_path, f"checkpoint-{total_idx}") , exist_ok=True)
                                torch.save(model, os.path.join(k_save_path, f"checkpoint-{total_idx}", "model.bin"))

                            if eval_average_loss < best_eval_loss:
                                os.makedirs( os.path.join(k_save_path, "best_loss") , exist_ok=True)
                                torch.save(model, os.path.join(k_save_path, "best_loss", "model.bin"))
                                best_eval_loss = eval_average_loss

                            if eval_average_f1 > best_eval_f1:
                                os.makedirs( os.path.join(k_save_path, "best_f1") , exist_ok=True)
                                torch.save(model, os.path.join(k_save_path, "best_f1", "model.bin"))
                                best_eval_f1 = eval_average_f1

                        else:
                            if args.checkpoint:
                                model.save_pretrained(os.path.join(k_save_path, f"checkpoint-{total_idx}"))

                            if eval_average_loss < best_eval_loss:
                                model.save_pretrained(os.path.join(k_save_path, "best_loss"))
                                best_eval_loss = eval_average_loss

                            if eval_average_f1 > best_eval_f1:
                                model.save_pretrained(os.path.join(k_save_path, "best_f1"))
                                best_eval_f1 = eval_average_f1

                        if args.wandb == "True":
                            wandb.log({
                                "step":total_idx,
                                "eval_loss":eval_average_loss,
                                "eval_f1":eval_average_f1,
                                "eval_acc":eval_average_acc,
                                "lr":optim.param_groups[0]["lr"]
                                })

                        if args.lr_scheduler:
                            scheduler.step(eval_average_loss) # ReduceLROnPlateau에는 loss 필요

                        print(f"[K_FOLD:({kfold_idx})][EVAL][loss:{eval_average_loss:4.2f} | auprc:{eval_total_auprc:4.2f} | ", end="")
                        print(f"micro_f1_score:{eval_average_f1:4.2f} | accuracy:{eval_average_acc:4.2f}]")

                    print("--------------------------------------------------------------------------")
            if args.wandb == "True":
                wandb.log({
                    "epoch":epoch+1,
                    "train_loss":average_loss,
                    "train_f1":average_f1,
                    "train_acc":average_acc,
                    "learning_rate": optim.param_groups[0]['lr']
                    })

        
        if args.wandb == "True":
            wandb.finish()


def main(args):
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
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default="cross_entropy") # 'cross_entropy', 'focal', 'label_smoothing', 'f1'
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--report_name', type=str)
    parser.add_argument('--project_name', type=str, default="salt_v3")
    parser.add_argument('--token_type', type=str, default="origin") # 'origin', 'entity', 'type_entity', 'sub_obj', 'special_entity'
    parser.add_argument('--wandb', type=str, default="True")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--sep_type', type=str, default='SEP') # SEP, ENT
    parser.add_argument('--lr_scheduler', type=str) # 'stepLR', 'reduceLR', 'cosine_anneal_warm', 'cosine_anneal', 'custom_cosine'
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--kfold_splits', type=int, default=5)
    parser.add_argument('--model_case', type=str, default='basic') # basic, electra
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--use_sub_obj', type=bool, default=False)

    args = parser.parse_args()
    main(args)
