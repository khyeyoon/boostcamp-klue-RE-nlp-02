import json
from gevent import config
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                    )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        prob = np.append(np.zeros([prob.shape[0], 1]), prob, axis=1)
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1) + 1

        output_pred.append(result)
        output_prob.append(prob)
    
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def inference_binary(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                    )
        logits = outputs[0]
        prob = [[1.] + [0.] * 29 for _ in range(logits.shape[0])] # F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        logits = softmax(logits)
        result = np.array((logits[:, 0]<0.6), dtype=int)
        # result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)
    
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    
    return origin_label

def load_test_dataset(dataset_dir, tokenizer, token_type, sep_type):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, token_type)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, sep_type)
    return test_dataset['id'], tokenized_test, test_label

def main(args, phase):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """

    test_data_path = "../dataset/test/test_data.csv"
    classified_test_data_path = "../dataset/test/classified_test_data.csv"
    output_file = './prediction/' + args.submission_name + ".csv"
    
    if not os.path.exists("./prediction"):
        os.mkdir("./prediction")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load tokenizer
    Tokenizer_NAME = os.path.join(args.model_dir, phase)
    # Tokenizer_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    print('tokenizer', tokenizer)

    ## load my model
    MODEL_NAME = os.path.join(args.model_dir, phase, args.best_path) # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    #print(model)

    with open(os.path.join(args.model_dir, phase, "model_config_parameters.json"), 'r') as f:
        model_config_parameters = json.load(f)
        token_type = model_config_parameters['token_type']
        sep_type = model_config_parameters['sep_type']
    #print(model_config_parameters)
    
    ## load test datset
    if phase == "no_RC":
        test_dataset_dir = test_data_path
    elif phase == "RC":
        test_dataset_dir = classified_test_data_path

    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, token_type, sep_type)

    Re_test_dataset = RE_Dataset(test_dataset ,test_label)

    data_idx = np.random.randint(0, 100)

    print(tokenizer.decode(Re_test_dataset[data_idx]['input_ids']))

    ## predict answer

    if phase == "no_RC":
        pred_answer, output_prob = inference_binary(model, Re_test_dataset, device) # model에서 class 추론

        df = pd.read_csv(test_dataset_dir)

        classified_answer = pd.DataFrame({"is_relation":pred_answer, "output_prob":output_prob})
        classified_df = pd.concat([df, classified_answer], axis=1)

        is_relation_df = classified_df[classified_df.is_relation == 1]
        is_relation_df.to_csv(classified_test_data_path, index=False)
        
        no_relation_df = classified_df[classified_df.is_relation == 0]

        output = pd.DataFrame({'id':no_relation_df.id, 'pred_label':num_to_label(no_relation_df.is_relation), 'probs':no_relation_df.output_prob,})

        output.to_csv(output_file, index=False)

        print(f"is_relation 데이터 {classified_test_data_path} 저장")
        print(classified_df.head())

    elif phase == "RC":
        pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
        is_relation_df = pd.read_csv(classified_test_data_path)

        pred_answer = num_to_label(pred_answer)

        old_df = pd.read_csv(output_file)
        new_df = pd.DataFrame({'id':is_relation_df.id, 'pred_label':pred_answer, 'probs':output_prob,})

        output = pd.concat([old_df, new_df]).sort_values('id')
        output.to_csv(output_file, index=False)

     # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    # output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})


    # output.to_csv(output_file, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--model_dir', type=str, default="./results")
    parser.add_argument('--best_path', type=str, default="best_loss")
    parser.add_argument('--submission_name', type=str, default="submission")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    print(args)
    main(args, phase="no_RC")
    main(args, phase="RC")
    